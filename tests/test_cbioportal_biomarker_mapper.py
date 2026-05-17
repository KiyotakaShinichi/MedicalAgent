import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.cbioportal_biomarker_mapper import _map_attributes


def test_cbioportal_mapper_groups_receptor_subtype_and_outcome_fields():
    attributes = [
        {"clinicalAttributeId": "ER_STATUS_BY_IHC", "displayName": "ER Status", "datatype": "STRING"},
        {"clinicalAttributeId": "PR_STATUS_BY_IHC", "displayName": "PR Status", "datatype": "STRING"},
        {"clinicalAttributeId": "HER2_STATUS", "displayName": "HER2 Status", "datatype": "STRING"},
        {"clinicalAttributeId": "PAM50_SUBTYPE", "displayName": "PAM50 subtype", "datatype": "STRING"},
        {"clinicalAttributeId": "OS_MONTHS", "displayName": "Overall survival months", "datatype": "NUMBER"},
        {"clinicalAttributeId": "DAYS_TO_DISTANT_RECURRENCE", "displayName": "Distant recurrence", "datatype": "NUMBER"},
    ]

    mapped = _map_attributes(attributes)

    assert mapped["er_status"][0]["id"] == "ER_STATUS_BY_IHC"
    assert mapped["pr_status"][0]["id"] == "PR_STATUS_BY_IHC"
    assert mapped["her2_status"][0]["id"] == "HER2_STATUS"
    assert mapped["subtype"][0]["id"] == "PAM50_SUBTYPE"
    assert mapped["survival"][0]["id"] == "OS_MONTHS"
    assert mapped["metastasis_recurrence"][0]["id"] == "DAYS_TO_DISTANT_RECURRENCE"
