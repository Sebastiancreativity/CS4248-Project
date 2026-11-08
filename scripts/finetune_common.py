"""Small, dependency-free integrity helpers for fine-tuning artifacts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_parallel(directory, name):
    src, ref = [(Path(directory) / f"{name}.{lang}").read_text(encoding="utf-8-sig").splitlines()
                for lang in ("zh", "en")]
    if not src or len(src) != len(ref) or any(not s.strip() for s in src + ref):
        raise ValueError(f"Invalid or empty parallel data: {name}")
    return src, ref


def load_run(run_dir):
    manifest = read_json(Path(run_dir) / "run.json")
    if manifest["fingerprint"] != fingerprint(manifest["spec"]):
        raise ValueError("Run manifest was modified; use a new run directory")
    for path, expected in manifest["spec"]["input_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Input changed after preparation: {path}. Use a new run directory.")
    for path, expected in manifest["spec"]["recipe_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Recipe changed after preparation: {path}. Use a new run directory.")
    return manifest


def adapter_record(adapter_dir, model_name):
    """Require completed training and match the actual saved adapter bytes."""
    adapter_dir = Path(adapter_dir)
    record = read_json(adapter_dir / "training_complete.json")
    if record.get("model") != model_name or record.get("global_step", 0) < 1:
        raise ValueError(f"No completed training for {model_name} in {adapter_dir}")
    for name, expected in record["files"].items():
        if Path(name).name != name or sha256(adapter_dir / name) != expected:
            raise ValueError(f"Changed or incomplete adapter: {adapter_dir / name}")
    if not {"adapter_config.json", "adapter_model.safetensors"} <= record["files"].keys():
        raise ValueError(f"Missing adapter weights in {adapter_dir}")
    return record


def generation_identity(model_name, adapter_dir, max_new_tokens, base_only=False):
    record = adapter_record(adapter_dir, model_name)
    return fingerprint({"model": model_name, "adapter": record, "base_only": base_only,
                        "max_new_tokens": max_new_tokens,
                        "code": {name: sha256(ROOT / "scripts" / name) for name in
                                 ("screen_model.py", "model_registry.py", "translation_cleaning.py")}})
