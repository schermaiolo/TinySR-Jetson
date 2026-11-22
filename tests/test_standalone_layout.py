"""Static checks for a self-contained TinySR exporter installation."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class StandaloneLayoutTests(unittest.TestCase):
    def test_exporter_does_not_reference_upstream_checkout(self):
        source = (ROOT / "tools/export_models.py").read_text()
        tree = ast.parse(source)
        opts = [value.value for value in ast.walk(tree)
                if isinstance(value, ast.Constant) and isinstance(value.value, str)]
        self.assertNotIn("--upstream", opts)
        self.assertIn("--checkpoint-root", opts)
        self.assertIn("--assets-root", opts)

    def test_all_model_sources_vendor_manifest(self):
        manifest = (ROOT / "scripts/import_upstream_sources.py").read_text()
        for filename in ("tinysd3.py", "tinysd3block.py", "autoencoder_tiny.py", "tinyvae.py"):
            self.assertIn(filename, manifest)


if __name__ == "__main__":
    unittest.main()
