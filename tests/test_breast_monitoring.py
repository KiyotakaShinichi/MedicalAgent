"""Compatibility collector for the split breast-monitoring integration suite."""

import unittest

from tests.breast_monitoring.environment import (
    configure_breast_monitoring_test_environment,
)

configure_breast_monitoring_test_environment()

from tests.breast_monitoring.test_imaging_and_data import ImagingAndDataTestsMixin  # noqa: E402
from tests.breast_monitoring.test_clinical_reports import ClinicalReportsTestsMixin  # noqa: E402
from tests.breast_monitoring.test_ml_operations import MLOperationsTestsMixin  # noqa: E402
from tests.breast_monitoring.test_rag_and_security import RAGAndSecurityTestsMixin  # noqa: E402
from tests.breast_monitoring.test_chat_routing import ChatRoutingTestsMixin  # noqa: E402
from tests.breast_monitoring.test_chat_safety_and_uploads import ChatSafetyAndUploadsTestsMixin  # noqa: E402
from tests.breast_monitoring.test_governance_and_domain import GovernanceAndDomainTestsMixin  # noqa: E402


class BreastMonitoringNLPTests(
    ImagingAndDataTestsMixin,
    ClinicalReportsTestsMixin,
    MLOperationsTestsMixin,
    RAGAndSecurityTestsMixin,
    ChatRoutingTestsMixin,
    ChatSafetyAndUploadsTestsMixin,
    GovernanceAndDomainTestsMixin,
    unittest.TestCase,
):
    """Backward-compatible aggregate preserving historical test node IDs."""


if __name__ == "__main__":
    unittest.main()
