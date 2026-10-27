"""Work around Windows crashes in file-backed PyTorch checkpoint storage.

Only the CPU checkpoint read is replaced. Tensor values and quantization settings
are preserved. Buffers are released when Transformers finishes each shard.
"""
import json
import mmap
import struct
import sys
import zipfile


class _BufferedTorchArchive:
    """ZIP record interface for the pinned PyTorch weights-only decoder.

    Read storage bytes through Python ZIP I/O rather than the native ZIP reader
    that crashed on ALMA's large Windows checkpoint shards.
    """

    def __init__(self, archive):
        self.archive = archive
        pickle_names = [name for name in archive.namelist()
                        if name == "data.pkl" or name.endswith("/data.pkl")]
        if len(pickle_names) != 1:
            raise ValueError("Expected one PyTorch data.pkl record")
        self.prefix = pickle_names[0][:-len("data.pkl")]
        self._mapped_file = None

    def has_record(self, name):
        try:
            self.archive.getinfo(self.prefix + name)
            return True
        except KeyError:
            return False

    def get_record(self, name):
        return self.archive.read(self.prefix + name)

    def get_storage_from_record(self, name, size, storage_type):
        import torch
        info = self.archive.getinfo(self.prefix + name)
        if info.file_size != size:
            raise OSError(f"Unexpected checkpoint storage size for {name}")
        if size == 0:
            return torch.empty(0, dtype=torch.uint8)
        # ALMA's ZIP records are stored (not compressed). Map the original
        # checkpoint and expose each tensor's byte range directly. Building a
        # bytearray for every tensor copied an entire ~10 GB shard into RAM and
        # could raise MemoryError on machines with limited free memory.
        if info.compress_type != zipfile.ZIP_STORED:
            raise OSError(f"Unsupported compressed checkpoint storage: {name}")
        if self._mapped_file is None:
            path = self.archive.filename
            with open(path, "rb") as handle:
                # Read-only mapping does not reserve pagefile commit for the
                # full checkpoint (ACCESS_COPY did on Windows). Transformers
                # only reads source tensors while casting/loading them.
                self._mapped_file = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        mapped = self._mapped_file
        header_offset = info.header_offset
        signature, = struct.unpack_from("<I", mapped, header_offset)
        if signature != 0x04034B50:
            raise OSError(f"Invalid ZIP storage header for {name}")
        name_length, extra_length = struct.unpack_from("<HH", mapped, header_offset + 26)
        data_offset = header_offset + 30 + name_length + extra_length
        if data_offset + size > len(mapped):
            raise OSError(f"Checkpoint storage extends past end of file: {name}")
        return torch.frombuffer(memoryview(mapped)[data_offset:data_offset + size],
                                dtype=torch.uint8)

    def serialization_id(self):
        return (self.get_record(".data/serialization_id").decode("ascii")
                if self.has_record(".data/serialization_id") else "")


def install_windows_checkpoint_reader():
    if sys.platform != "win32":
        return

    import torch
    from safetensors import safe_open
    from transformers import modeling_utils

    original = modeling_utils.load_state_dict
    if getattr(original, "_buffered_windows_reader", False):
        return

    def buffered_load(checkpoint_file, is_quantized=False, map_location="cpu", weights_only=True):
        if str(map_location) == "meta":
            return original(checkpoint_file, is_quantized=is_quantized,
                            map_location=map_location, weights_only=weights_only)
        if not str(checkpoint_file).endswith(".safetensors"):
            # ALMA ships .bin shards. Transformers normally enables mmap for
            # these too; the later dtype conversion then reads mapped storage.
            # Retain Transformers' security check and restricted weights loader.
            if weights_only:
                modeling_utils.check_torch_load_is_safe()
                if zipfile.is_zipfile(checkpoint_file):
                    from torch import _weights_only_unpickler
                    with zipfile.ZipFile(checkpoint_file) as archive:
                        return torch.serialization._load(
                            _BufferedTorchArchive(archive), map_location or "cpu",
                            _weights_only_unpickler, weights_only=True, encoding="utf-8")
            return torch.load(checkpoint_file, map_location=map_location or "cpu",
                              weights_only=weights_only, mmap=False)

        # Safetensors validates the header, shapes, dtype sizes, and data offsets.
        # Avoid get_tensor(): its PyTorch file-backed storage slice is the path
        # that produced the native Windows access violation on this machine.
        with safe_open(checkpoint_file, framework="np") as archive:
            metadata = archive.metadata()
            keys = list(archive.keys())
        if metadata is not None and metadata.get("format") not in ("pt", "tf", "flax", "mlx"):
            raise OSError(f"Invalid checkpoint format metadata: {checkpoint_file}")

        state_dict = {}
        with open(checkpoint_file, "rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_size))
            data_start = 8 + header_size
        # Safetensors payloads are contiguous and uncompressed. Mapping them
        # read-only avoids copying an entire large checkpoint (MADLAD is ~12GB)
        # into Python bytearrays before the model is moved to the GPU.
        with open(checkpoint_file, "rb") as handle:
            mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        if data_start > len(mapped):
            mapped.close()
            raise OSError(f"Invalid Safetensors header size: {checkpoint_file}")
        for key in keys:
            info = header[key]
            dtype = modeling_utils.str_to_torch_dtype[info["dtype"]]
            start, end = info["data_offsets"]
            if start == end:
                tensor = torch.empty(info["shape"], dtype=dtype)
            else:
                if start < 0 or end < start or data_start + end > len(mapped):
                    raise OSError(f"Incomplete tensor {key} in {checkpoint_file}")
                # The tensor retains its memoryview, which keeps the mmap
                # alive after this loader returns. Values remain read-only
                # source weights until Transformers copies them to CUDA.
                tensor = torch.frombuffer(
                    memoryview(mapped)[data_start + start:data_start + end], dtype=dtype
                ).reshape(info["shape"])
            state_dict[key] = tensor
        return state_dict

    buffered_load._buffered_windows_reader = True
    modeling_utils.load_state_dict = buffered_load
    print("Using Windows checkpoint reader (Safetensors and .bin ZIP storage memory-mapped).", flush=True)
