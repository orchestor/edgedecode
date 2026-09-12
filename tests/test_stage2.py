import json
import unittest
from pathlib import Path

from edgedecode.stage2_analysis import summarize


ROOT = Path(__file__).resolve().parents[1]


class Stage2EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = json.loads((ROOT / "results/m4-stage2/benchmark.json").read_text())
        cls.quality = json.loads((ROOT / "results/m4-stage2/quality.json").read_text())
        cls.config = json.loads((ROOT / "configs/stage2.json").read_text())
        cls.summary = summarize(cls.benchmark, cls.quality, cls.config["models"])

    def test_complete_balanced_matrix(self):
        rows = self.benchmark["rows"]
        self.assertEqual(len(rows), 18)
        self.assertEqual({r["model_label"] for r in rows}, {"F16", "Q8_0", "Q4_K_M"})
        self.assertEqual({r["requested_backend"] for r in rows}, {"cpu", "metal"})
        self.assertTrue(all(len(r["samples_ts"]) == 7 for r in rows))
        self.assertTrue(all(r["n_gpu_layers"] == 0 for r in rows if r["requested_backend"] == "cpu"))
        self.assertTrue(all(r["n_gpu_layers"] > 0 for r in rows if r["requested_backend"] == "metal"))

    def test_quality_protocol_is_matched(self):
        self.assertEqual({r["chunks"] for r in self.quality["perplexity"]}, {32})
        self.assertEqual({r["context"] for r in self.quality["perplexity"]}, {512})
        self.assertEqual({r["tasks"] for r in self.quality["hellaswag"]}, {200})
        self.assertEqual({r["seed"] for r in self.quality["hellaswag"]}, {1234})

    def test_derived_decision_numbers(self):
        perf = self.summary["performance"]
        self.assertAlmostEqual(perf["cpu"]["tg128"]["Q4_K_M"]["speedup_vs_f16"], 1.947, places=3)
        self.assertAlmostEqual(perf["metal"]["tg128"]["Q4_K_M"]["speedup_vs_f16"], 2.253, places=3)
        self.assertAlmostEqual(self.summary["models"]["Q4_K_M"]["perplexity_delta_percent_vs_f16"], 3.94, places=2)

    def test_manifest_avoids_unique_machine_identifiers(self):
        manifest = (ROOT / "results/m4-stage2/manifest.json").read_text().lower()
        for field in ("serial_number", "hardware_uuid", "provisioning_udid"):
            self.assertNotIn(field, manifest)

    def test_recorded_commands_are_checkout_independent(self):
        payloads = [self.benchmark, self.quality]
        serialized = json.dumps(payloads)
        self.assertNotIn(str(ROOT), serialized)
        self.assertNotIn("/Users/", serialized)


if __name__ == "__main__":
    unittest.main()
