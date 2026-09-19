import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gpt_from_scratch.checkpoint import check_tokenizer, load_training
from gpt_from_scratch.tokenizer import CharTokenizer, load_tokenizer

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def run_cli(self, *args, success=True):
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=60)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def prepare(self, root):
        data = Path(root) / "data"
        self.run_cli("prepare_data.py", "--input", ROOT / "examples/tiny_dialogues.jsonl",
                     "--out-dir", data, "--seq-len", 8, "--val-frac", .2, "--assistant-loss-only")
        return data

    def test_prepare_train_resume_sample_exactly(self):
        with tempfile.TemporaryDirectory() as root:
            data = self.prepare(root)
            full, resumed = Path(root) / "full", Path(root) / "resumed"
            args = ["train.py", "--data-dir", data, "--steps", 6, "--batch-size", 2,
                    "--d-model", 8, "--n-layers", 1, "--n-heads", 2,
                    "--warmup-steps", 1, "--eval-every", 3, "--eval-iters", 2,
                    "--save-every", 3, "--device", "cpu"]
            self.run_cli(*args, "--checkpoint-dir", full)
            self.run_cli(*args, "--checkpoint-dir", resumed, "--stop-after", 2)
            self.run_cli(*args, "--checkpoint-dir", resumed, "--resume", resumed / "latest.npz")
            for filename in ("latest.npz", "best.npz"):
                with np.load(full / filename) as a, np.load(resumed / filename) as b:
                    self.assertEqual(set(a.files), set(b.files))
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
            sample = ["sample.py", "--data-dir", data, "--checkpoint", resumed / "latest.npz",
                      "--prompt", "unseen🙂", "--max-new-tokens", 5, "--seed", 7, "--device", "cpu"]
            first, second = self.run_cli(*sample), self.run_cli(*sample)
            self.assertTrue(first.stdout.startswith("unseen🙂"))
            self.assertEqual(first.stdout, second.stdout)
            # Fine-tuning inherits architecture and starts a fresh optimizer.
            fine = Path(root) / "fine"
            self.run_cli("train.py", "--data-dir", data, "--checkpoint-dir", fine,
                         "--init-from", full / "latest.npz", "--steps", 1,
                         "--batch-size", 1, "--eval-iters", 1)
            with np.load(fine / "latest.npz") as ckpt:
                self.assertEqual(int(ckpt["optim.t"]), 1)
                self.assertEqual(int(ckpt["extra.d_model"]), 8)
            result = self.run_cli(*args, "--checkpoint-dir", resumed, "--resume", resumed / "latest.npz",
                                  "--lr", .001, success=False)
            self.assertIn("original training configuration", result.stderr)
            tok = load_tokenizer(data / "tokenizer.json")
            stoi = dict(tok.stoi)
            ordinary = [key for key in stoi if key not in ("<pad>", "<unk>", "<bos>", "<eos>")]
            a, b = ordinary[:2]
            stoi[a], stoi[b] = stoi[b], stoi[a]
            with self.assertRaisesRegex(ValueError, "mismatch"):
                check_tokenizer(full / "latest.npz", CharTokenizer(stoi))

    def test_preparation_split_and_train_only_vocabulary(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source.jsonl"
            with source.open("w", encoding="utf-8") as f:
                for i in range(10):
                    doc = {"conversations": [{"role": "user", "content": "问题"},
                                             {"role": "assistant", "content": chr(0x4e50 + i) * 12}]}
                    f.write(json.dumps(doc) + "\n\n")
                    f.write(json.dumps(doc) + "\n")
            data = Path(root) / "data"
            self.run_cli("prepare_data.py", "--input", source, "--out-dir", data,
                         "--seq-len", 8, "--val-frac", .3, "--assistant-loss-only")
            meta = json.loads((data / "meta.json").read_text())
            tok = load_tokenizer(data / "tokenizer.json")
            self.assertEqual(meta["train_docs"], 14)
            self.assertEqual(meta["val_docs"], 6)
            self.assertEqual(sum(chr(0x4e50 + i) in tok.stoi for i in range(10)), 7)
            train = np.fromfile(data / "train.bin", dtype=meta["dtype"])
            val = np.fromfile(data / "val.bin", dtype=meta["dtype"])
            self.assertEqual(int(train[0]), tok.bos_id)
            self.assertEqual(int(val[0]), tok.bos_id)
            self.assertEqual(int(train[-1]), tok.eos_id)
            self.assertEqual(int(val[-1]), tok.eos_id)
            self.assertIn(tok.unk_id, val)

    def test_subword_cli_and_missing_mask_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            data = Path(root) / "data"
            self.run_cli("prepare_data.py", "--input", ROOT / "examples/tiny_dialogues.jsonl",
                         "--out-dir", data, "--seq-len", 8, "--tokenizer-type", "subword",
                         "--assistant-loss-only")
            self.run_cli("train.py", "--data-dir", data, "--checkpoint-dir", Path(root) / "ckpt",
                         "--steps", 1, "--batch-size", 1, "--d-model", 8,
                         "--n-layers", 1, "--n-heads", 2, "--eval-iters", 1)
            (data / "train_mask.bin").unlink()
            failed = self.run_cli("train.py", "--data-dir", data, "--steps", 1, success=False)
            self.assertIn("train_mask.bin", failed.stderr)

    def test_too_small_data_leaves_output_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            data = Path(root) / "data"
            data.mkdir()
            marker = data / "meta.json"
            marker.write_text("preserve")
            result = self.run_cli("prepare_data.py", "--input", ROOT / "examples/tiny_dialogues.jsonl",
                                  "--out-dir", data, "--seq-len", 100000, success=False)
            self.assertIn("seq_len + 1", result.stderr)
            self.assertEqual(marker.read_text(), "preserve")

    def test_builders_create_output_directories_and_filter_preserves_input(self):
        with tempfile.TemporaryDirectory() as root:
            math_file = Path(root) / "math/nested/samples.jsonl"
            self.run_cli("build_math_drill.py", "--output", math_file, "--max-add", 2,
                         "--max-mul", 2, "--repeat", 1)
            self.assertEqual(len(math_file.read_text().splitlines()), 20)
            filtered = Path(root) / "filtered/samples.jsonl"
            self.run_cli("filter_dataset.py", "--input", math_file, "--output", filtered, "--min-answer-chars", 1)
            original = math_file.read_bytes()
            self.run_cli("filter_dataset.py", "--input", math_file, "--output", math_file, success=False)
            self.assertEqual(original, math_file.read_bytes())
            self.run_cli("build_curriculum.py", "--input", math_file, "--output", Path(root) / "curriculum/samples.jsonl",
                         "--seed-repeat", 1, "--max-source", 2, "--max-general", 1, "--max-topic", 1)


if __name__ == "__main__":
    unittest.main()
