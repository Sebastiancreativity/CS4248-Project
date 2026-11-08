import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import finetune_pipeline as pipeline
from finetune_common import adapter_record, fingerprint, load_run, sha256, write_json
from finetune_data import TranslationCollator, encode_pair
from model_registry import MODELS, TEAM_MODELS, build_prompt


class Tokenizer:
    chat_template = "chat"
    eos_token = "~"
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        prefix = "USER:" + messages[0]["content"] + " ASSISTANT:"
        return prefix if len(messages) == 1 else prefix + messages[1]["content"] + "~"
    def __call__(self, text=None, text_target=None, add_special_tokens=False):
        return {"input_ids": ([999] if text_target is not None else []) +
                [ord(c) for c in (text_target if text_target is not None else text)]}
    def pad(self, features, **kwargs):
        return {"input_ids": [x["input_ids"] for x in features]}


class TrainingDataTests(unittest.TestCase):
    def test_all_decoder_models_mask_source_and_preserve_english_and_eos(self):
        tok = Tokenizer()
        for model in TEAM_MODELS:
            cfg = MODELS[model]
            if cfg.get("seq2seq"):
                continue
            encoded = encode_pair(tok, cfg, "中文", "English")
            prompt = tok(build_prompt(tok, cfg, "中文"))["input_ids"]
            self.assertEqual(encoded["labels"][:len(prompt)], [-100] * len(prompt))
            self.assertEqual(encoded["labels"][len(prompt):], list(map(ord, "English~")))
            self.assertEqual(encoded["input_ids"], prompt + list(map(ord, "English~")))

    def test_seq2seq_models_keep_target_out_of_encoder_input(self):
        for model in ("nllb-1.3b", "madlad400-3b"):
            result = encode_pair(Tokenizer(), MODELS[model], "中文", "English")
            expected = ("<2en> " if model.startswith("madlad") else "") + "中文"
            self.assertEqual(result["input_ids"], list(map(ord, expected)))
            self.assertEqual(result["labels"], [999] + list(map(ord, "English")))

    def test_overlength_is_dropped_instead_of_training_mismatched_halves(self):
        self.assertIsNone(encode_pair(Tokenizer(), {}, "中文", "English", max_source_length=1))
        self.assertIsNone(encode_pair(Tokenizer(), {}, "中文", "English", max_target_length=1))

    def test_padding_preserves_eos_label_and_ignores_only_added_padding(self):
        with patch.dict(sys.modules, {"torch": SimpleNamespace(tensor=lambda rows, **kw: rows, long="long")}):
            result = TranslationCollator(Tokenizer(), SimpleNamespace())([
                {"input_ids": [1, 2, 9], "labels": [-100, 2, 9]},
                {"input_ids": [1, 9], "labels": [-100, 9]}])
        self.assertEqual(result["labels"][0], [-100, 2, 9, -100, -100, -100, -100, -100])
        self.assertEqual(result["labels"][1][1], 9)
        self.assertNotIn("decoder_input_ids", result)


class IntegrityTests(unittest.TestCase):
    def test_finetuned_cache_rejects_pretrained_or_different_adapter_outputs(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("cache_check_runner", ROOT / "scripts/run_ensembles.py")
        runner = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"cache_paths": SimpleNamespace(COMET_CACHE=Path(".")),
                                     "sacrebleu": SimpleNamespace()}):
            spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            (path / "dev_train20.hyp").write_text("English\n")
            (path / "dev_train20.raw.jsonl").write_text(json.dumps({"src": "source", "raw": "English"}) + "\n")
            self.assertTrue(runner.valid_cache(path, "dev_train20", ["source"]))
            self.assertFalse(runner.valid_cache(path, "dev_train20", ["source"], "adapter-A"))
            write_json(path / "dev_train20.generation.json", {"identity": "adapter-A"})
            self.assertTrue(runner.valid_cache(path, "dev_train20", ["source"], "adapter-A"))
            self.assertFalse(runner.valid_cache(path, "dev_train20", ["source"], "adapter-B"))

    def test_corrupt_or_incomplete_adapters_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                (path / name).write_text("example")
            record = {"model": "example", "global_step": 1,
                      "files": {p.name: sha256(p) for p in path.iterdir()}}
            write_json(path / "training_complete.json", record)
            self.assertEqual(adapter_record(path, "example"), record)
            with self.assertRaises(ValueError):
                adapter_record(path, "other")
            (path / "adapter_model.safetensors").write_text("modified")
            with self.assertRaises(ValueError):
                adapter_record(path, "example")

    def test_modified_data_invalidates_prepared_run(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            data = path / "train.zh"; data.write_text("source")
            spec = {"input_sha256": {str(data): sha256(data)}, "recipe_sha256": {}}
            write_json(path / "run.json", {"spec": spec, "fingerprint": fingerprint(spec)})
            load_run(path)
            data.write_text("changed")
            with self.assertRaises(ValueError):
                load_run(path)

    def test_model_selection_refuses_test_scores_and_can_keep_pretrained(self):
        rows = [dict(dev="dev_train20", selector=s, comet22="90", bleu="20", chrfpp="40", variant=v)
                for v in pipeline.VARIANTS for s in pipeline.SELECTORS]
        self.assertEqual(pipeline.choose_on_dev(rows)["primary"]["variant"], "pretrained")
        rows[-1]["comet22"] = "91"
        self.assertEqual(pipeline.choose_on_dev(rows)["primary"]["variant"], "finetuned")
        rows[0]["dev"] = "test_wmt22"
        with self.assertRaises(ValueError):
            pipeline.choose_on_dev(rows)

    def test_before_after_requires_matched_pairs_and_reports_negative_changes(self):
        rows = [dict(model="a", variant=v, comet22=str(n), bleu=str(n), chrfpp=str(n), bertscore_f1=str(n))
                for v, n in [("pretrained", 85), ("finetuned", 84)]]
        self.assertEqual(pipeline.paired_deltas(rows, ("model",))[0]["delta_comet22"], -1)
        with self.assertRaises(ValueError):
            pipeline.paired_deltas(rows[:1], ("model",))


class PipelineTests(unittest.TestCase):
    def test_dev_freezes_choice_before_all_test_evaluations(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); split = root / "split"; split.mkdir()
            for name in ("train", "dev_train20", "test_train10"):
                for lang in ("zh", "en"):
                    (split / f"{name}.{lang}").write_text(f"{name}-{lang}\n")
            for name in ("tatoeba.zh", "tatoeba.en", "wmttest2022.zh", "wmttest2022.AnnA.en"):
                (root / name).write_text(f"{name}\n")
            settings = {"generation_batch_size": 4, "max_new_tokens": 512, "metric_batch_size": 16}
            run = {"fingerprint": "run", "spec": {"split_dir": str(split),
                   "counts": {"dev_train20": 1}, "evaluation": settings}}
            assignments = pipeline.read_assignments(ROOT, None)
            commands = []
            def execute(command):
                command = list(map(str, command)); commands.append(command)
                script = Path(command[1]).name
                def arg(flag):
                    return command[command.index(flag) + 1]
                def datasets():
                    pos = command.index("--datasets") + 1
                    end = next((i for i in range(pos, len(command)) if command[i].startswith("--")), len(command))
                    return command[pos:end]
                if script == "score_finetuned_models.py":
                    rows = [dict(variant=arg("--variant"), model=m, dataset=ds, n=1,
                                 comet22=85, bleu=30, chrfpp=50, bertscore_f1=80)
                            for m in TEAM_MODELS for ds in datasets()]
                    path = root / "evaluation" / arg("--stage") / arg("--variant") / "individual" / "scores.csv"
                    pipeline.save_csv(path, rows)
                elif script == "run_ensembles.py":
                    assignment = next(a for a in assignments if a["owner"] == arg("--owner"))
                    rows = [dict(experiment=i, dev=ds, selector=s, models=" + ".join(trio), n=1,
                                 comet22=(90 if "--base-only" in command else 89), bleu=30, chrfpp=50, bertscore_f1=80)
                            for i, trio in enumerate(assignment["combinations"], 1)
                            for ds in datasets() for s in pipeline.SELECTORS]
                    pipeline.save_csv(Path(arg("--results-dir")) / arg("--owner") / "scores_comparison.csv", rows)
            with patch.multiple(pipeline, ROOT=root, load_run=lambda _: run,
                                adapter_record=lambda *a: {"run_fingerprint": "run"},
                                generation_identity=lambda *a: "id", execute=execute,
                                read_assignments=lambda *a: assignments), patch.dict(sys.modules, {
                                    "run_ensembles": SimpleNamespace(valid_cache=lambda *a: False)}):
                with self.assertRaises(FileNotFoundError):
                    pipeline.evaluate_locked(root, "test")
                self.assertEqual(commands, [])
                pipeline.evaluate_locked(root, "dev")
                chosen = json.loads((root / "selection.json").read_text())
                self.assertEqual(chosen["primary"]["variant"], "pretrained")
                self.assertEqual(len(pipeline.read_csv(root / "evaluation/dev/scores_comparison.csv")), 240)
                before = (root / "selection.json").read_bytes()
                pipeline.evaluate_locked(root, "test")
                self.assertEqual((root / "selection.json").read_bytes(), before)
                self.assertEqual(len(pipeline.read_csv(root / "evaluation/test/scores_comparison.csv")), 720)
                selected = pipeline.read_csv(root / "evaluation/test/final_selected_scores.csv")
                self.assertEqual(len(selected), 3)
                self.assertEqual({r["dev"] for r in selected}, {"test_wmt22", "test_tatoeba", "test_train10"})
                self.assertEqual({r["variant"] for r in selected}, {"pretrained"})
                screen = [c for c in commands if Path(c[1]).name == "screen_model.py"]
                self.assertEqual(len(screen), 32)  # eight models, two variants, two stages
                self.assertTrue(all("--adapter-dir" in c for c in screen))
                chosen["primary"]["comet22"] = "100"
                write_json(root / "selection.json", chosen)
                with self.assertRaises(ValueError):
                    pipeline.evaluate_locked(root, "test")

    def test_source_overlap_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for split in ("train", "dev_train20", "test_train10"):
                for lang in ("zh", "en"):
                    (root / f"{split}.{lang}").write_text(split)
            for name in ("tatoeba.zh", "wmttest2022.zh"):
                (root / name).write_text("official")
            pipeline.validate_disjoint(root, root)
            (root / "dev_train20.zh").write_text("ＴＲＡＩＮ", encoding="utf-8")
            with self.assertRaises(ValueError):
                pipeline.validate_disjoint(root, root)
