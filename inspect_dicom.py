import argparse
import json

from backend.services.dicom_inspector import inspect_dicom_tree


def main():
    parser = argparse.ArgumentParser(description="Inspect local QIN-BREAST-02 DICOM folders.")
    parser.add_argument("root_path", help="Root DICOM folder path")
    parser.add_argument("--patient-id", default=None, help="Optional patient ID filter")
    parser.add_argument("--max-files", type=int, default=None, help="Optional file scan limit")
    args = parser.parse_args()

    result = inspect_dicom_tree(
        root_path=args.root_path,
        patient_id=args.patient_id,
        max_files=args.max_files,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
