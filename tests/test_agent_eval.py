import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class AgentEvalConfigTests(unittest.TestCase):
    def test_qwen3_agent_config_exists(self):
        cfg = json.loads((ROOT / "configs/qwen3_agent.json").read_text())
        self.assertEqual(cfg["model_family"], "Qwen2.5-7B")
        self.assertIn("agent_tasks", cfg)
        self.assertGreater(len(cfg["agent_tasks"]), 0)
        self.assertIn("tool_use", cfg["agent_tasks"][0]["tags"])

    def test_agent_eval_script_exists(self):
        script = ROOT / "scripts" / "run_qwen3_agent_eval.py"
        self.assertTrue(script.exists())
        text = script.read_text()
        self.assertIn("agent_tasks", text)
        self.assertIn("Qwen2.5-7B", text)


if __name__ == "__main__":
    unittest.main()
