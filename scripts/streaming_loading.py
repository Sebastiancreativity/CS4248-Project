"""Load a single Safetensors checkpoint with bounded CPU staging memory."""
import json
import math

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from huggingface_hub import hf_hub_download
from transformers import AutoConfig


@torch.no_grad()
def load_streaming_safetensors(model_id, model_class, device="cuda", dtype=torch.bfloat16):
    """Stage 16 MiB at a time; cast source weights on the GPU, not the CPU.

    This bypasses Transformers' native Safetensors slicing path on Windows.
    The complete model still needs to fit in GPU memory for inference.
    """
    path = hf_hub_download(model_id, "model.safetensors")
    config = AutoConfig.from_pretrained(model_id)
    with init_empty_weights():
        model = model_class.from_config(config, dtype=dtype)
    model.tie_weights()
    # Save tied parameter names before replacing meta tensors. MADLAD stores
    # decoder.embed_tokens.weight for the shared encoder/decoder embedding.
    aliases = {}
    parameters = dict(model.named_parameters(remove_duplicate=False))
    for name, param in parameters.items():
        aliases.setdefault(id(param), []).append(name)

    types = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
             "F64": torch.float64, "I64": torch.int64, "I32": torch.int32,
             "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}
    chunk_bytes = 16 * 1024**2
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise OSError("Incomplete Safetensors header")
        header_size = int.from_bytes(prefix, "little")
        if not 0 < header_size <= 64 * 1024**2:
            raise OSError("Invalid Safetensors header size")
        header = json.loads(handle.read(header_size))
        data_start = 8 + header_size
        file_size = handle.seek(0, 2)
        entries = [(name, info) for name, info in header.items() if name != "__metadata__"]
        print(f"Streaming {len(entries)} tensors to {device} (16 MiB CPU buffer; {dtype}).", flush=True)
        for index, (name, info) in enumerate(entries, 1):
            if name not in parameters:
                raise ValueError(f"Unexpected checkpoint parameter: {name}")
            source_dtype = types[info["dtype"]]
            element_size = torch.empty((), dtype=source_dtype).element_size()
            shape = info["shape"]
            if tuple(shape) != tuple(parameters[name].shape):
                raise ValueError(f"Checkpoint shape mismatch: {name}")
            start, end = info["data_offsets"]
            if start < 0 or end - start != math.prod(shape) * element_size or data_start + end > file_size:
                raise OSError(f"Invalid checkpoint byte range: {name}")
            target_dtype = dtype if source_dtype.is_floating_point else source_dtype
            target = torch.empty(shape, device=device, dtype=target_dtype)
            flat = target.view(-1)
            handle.seek(data_start + start)
            remaining, offset = end - start, 0
            while remaining:
                count = min(chunk_bytes, remaining)
                buffer = bytearray(count)
                if handle.readinto(buffer) != count:
                    raise OSError(f"Incomplete checkpoint tensor: {name}")
                cpu_chunk = torch.frombuffer(buffer, dtype=source_dtype)
                # Transfer the original dtype first so BF16 conversion uses
                # CUDA, avoiding the crashing native CPU conversion path.
                gpu_chunk = cpu_chunk.to(device)
                flat[offset:offset + cpu_chunk.numel()].copy_(gpu_chunk)
                offset += cpu_chunk.numel()
                remaining -= count
                del gpu_chunk, cpu_chunk, buffer
            for alias in aliases[id(parameters[name])]:
                set_module_tensor_to_device(model, alias, device, value=target, dtype=target_dtype)
            del flat, target
            if index % 25 == 0 or index == len(entries):
                print(f"  Loaded {index}/{len(entries)} tensors", flush=True)
    model.tie_weights()
    missing = [name for name, param in model.named_parameters() if param.is_meta]
    if missing:
        raise ValueError(f"Checkpoint left parameters unloaded: {missing}")
    return model.to(device)
