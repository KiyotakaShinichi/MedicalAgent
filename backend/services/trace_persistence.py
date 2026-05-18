"""Trace facade for RAG telemetry.

The DB write still lives inside ``agent_rag._store_rag_evaluation_log`` to
avoid a risky circular refactor in this stabilization pass. New code can rely
on the trace-envelope builder here while the persistence move remains a small,
explicit future refactor.
"""

from backend.services.agent_trace import _trace, build_pipeline_trace  # noqa: F401

__all__ = ["_trace", "build_pipeline_trace"]
