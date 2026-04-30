import argparse
import json

from backend.services.breastdcedl_inspector import inspect_breastdcedl_dataset


def main():
    parser = argparse.ArgumentParser(description="Inspect a local BreastDCEDL archive or extracted folder.")
    parser.add_argument("path", help="Path to BreastDCEDL_spy1.zip, tar.gz, or extracted folder")
    args = parser.parse_args()

    print(json.dumps(inspect_breastdcedl_dataset(args.path), indent=2))


if __name__ == "__main__":
    main()
