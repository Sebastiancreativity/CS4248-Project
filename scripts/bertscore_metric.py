"""Reference-based BERTScore for English MT outputs, including saved .hyp files.

The encoder is loaded lazily and reused. CPU is the default so COMET can keep
using the GPU; pass --bertscore-device cuda when GPU memory is available.
"""
import argparse
import json
from pathlib import Path

DEFAULT_MODEL = "microsoft/deberta-large-mnli"


def add_arguments(parser):
    parser.add_argument("--bertscore", action="store_true", help="also report reference-based BERTScore")
    parser.add_argument("--bertscore-model", default=DEFAULT_MODEL,
                        help="Hugging Face encoder supported by bert-score")
    parser.add_argument("--bertscore-device", default="cpu", help="cpu (default), cuda, or cuda:0")
    parser.add_argument("--bertscore-batch-size", type=int, default=16)


def from_args(args):
    if not args.bertscore:
        return None
    return EnglishBERTScore(args.bertscore_model, args.bertscore_device, args.bertscore_batch_size)


class EnglishBERTScore:
    def __init__(self, model=DEFAULT_MODEL, device="cpu", batch_size=16):
        if batch_size < 1:
            raise ValueError("BERTScore batch size must be positive")
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self._scorer = None

    def score(self, hyp, ref):
        if len(hyp) != len(ref):
            raise ValueError("BERTScore hypothesis/reference line counts must match")
        if not hyp:
            raise ValueError("BERTScore requires at least one hypothesis/reference pair")
        if any(not text.strip() for text in ref):
            raise ValueError("BERTScore requires a nonempty reference for every hypothesis")
        if self._scorer is None:
            # Configure the project's download cache before Hugging Face imports.
            import cache_paths  # noqa: F401
            try:
                from bert_score import BERTScorer
            except ImportError as exc:
                raise RuntimeError("Install BERTScore with: python -m pip install bert-score==0.3.13") from exc
            self._scorer = BERTScorer(
                model_type=self.model, lang="en", device=self.device,
                idf=False, rescale_with_baseline=False, use_fast_tokenizer=False,
            )
        # Keep empty hypotheses in the average: failed translations must count.
        p, r, f1 = self._scorer.score(hyp, ref, batch_size=self.batch_size)
        return {
            "bertscore_precision": round(100 * p.mean().item(), 2),
            "bertscore_recall": round(100 * r.mean().item(), 2),
            "bertscore_f1": round(100 * f1.mean().item(), 2),
            "bertscore_hash": self._scorer.hash,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hyp", required=True, type=Path)
    parser.add_argument("--ref", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="optional JSON file for the scores")
    add_arguments(parser)
    args = parser.parse_args()
    hyp = args.hyp.read_text(encoding="utf-8").splitlines()
    ref = args.ref.read_text(encoding="utf-8").splitlines()
    scorer = EnglishBERTScore(args.bertscore_model, args.bertscore_device, args.bertscore_batch_size)
    result = {"n": len(hyp), **scorer.score(hyp, ref)}
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
