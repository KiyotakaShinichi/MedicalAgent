import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.services.imaging_baseline_experiments import build_ct_lesion_workflow_report, run_ultrasound_baseline
from backend.services.public_imaging_datasets import build_public_imaging_manifest


class PublicImagingServiceTests(unittest.TestCase):
    def test_public_imaging_manifest_reports_missing_datasets_honestly(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "manifest.json"
            result = build_public_imaging_manifest(dataset_root=str(Path(tmp) / "Datasets"), output_path=str(output))

            self.assertEqual(result["status"], "datasets_not_downloaded")
            self.assertEqual(result["available_dataset_count"], 0)
            self.assertTrue(output.exists())

    def test_ultrasound_baseline_runs_on_minimal_public_dataset_shape(self):
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"PIL unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "BUSI"
            for label, offset in [("benign", 40), ("malignant", 160)]:
                folder = root / label
                folder.mkdir(parents=True)
                for index in range(12):
                    arr = np.full((32, 32), offset + index, dtype=np.uint8)
                    Image.fromarray(arr).save(folder / f"{label}_{index}.png")

            output = Path(tmp) / "metrics.json"
            predictions = Path(tmp) / "predictions.csv"
            result = run_ultrasound_baseline(
                dataset_root=str(root),
                output_path=str(output),
                predictions_path=str(predictions),
                max_images=40,
            )

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["image_count"], 24)
            self.assertIn(result["best_model"], result["models"])
            self.assertTrue(output.exists())
            self.assertTrue(predictions.exists())

    def test_ct_workflow_report_counts_metadata_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "DeepLesion"
            root.mkdir()
            (root / "annotations.csv").write_text("id,x,y\n1,2,3\n2,4,5\n", encoding="utf-8")
            (root / "slice001.png").write_bytes(b"placeholder")

            output = Path(tmp) / "ct_report.json"
            result = build_ct_lesion_workflow_report(dataset_root=str(root), output_path=str(output))

            self.assertEqual(result["status"], "workflow_ready")
            self.assertEqual(result["metadata_file_count"], 1)
            self.assertEqual(result["estimated_annotation_rows"], 2)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
