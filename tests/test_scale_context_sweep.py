import unittest

from scripts.run_scale_context_sweep import build_case_table


class ScaleContextSweepTests(unittest.TestCase):
    def test_build_case_table_has_expected_matrix(self):
        cases = build_case_table()
        self.assertGreater(len(cases), 0)

        expected = {
            ("qwen2.5-1.5b", "cpu", "prefill", 128, 1),
            ("qwen2.5-1.5b", "metal", "prefill", 2048, 16),
            ("qwen2.5-7b", "cpu", "decode", 128, 1),
            ("qwen2.5-7b", "metal", "decode", 128, 8),
        }
        ids = {(c["model"], c["backend"], c["phase"], c["context"], c["batch"]) for c in cases}
        for item in expected:
            self.assertIn(item, ids)

        self.assertEqual(len([c for c in cases if c["phase"] == "prefill"]), 54)
        self.assertEqual(len([c for c in cases if c["phase"] == "decode"]), 18)


if __name__ == "__main__":
    unittest.main()
