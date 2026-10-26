"""Keep downloaded model files on the data drive.

Set CS4248_MODEL_CACHE before launching Python to use a different location.
"""
import os
from pathlib import Path


CACHE_ROOT = Path(os.environ.get("CS4248_MODEL_CACHE", r"E:\huggingface"))
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
COMET_CACHE = CACHE_ROOT / "comet"
COMET_CACHE.mkdir(parents=True, exist_ok=True)

# These must be assigned before importing transformers, huggingface_hub, or comet.
os.environ["HF_HOME"] = str(CACHE_ROOT)
os.environ["HF_HUB_CACHE"] = str(CACHE_ROOT / "hub")
os.environ["HF_DATASETS_CACHE"] = str(CACHE_ROOT / "datasets")
os.environ["TORCH_HOME"] = str(CACHE_ROOT / "torch")
