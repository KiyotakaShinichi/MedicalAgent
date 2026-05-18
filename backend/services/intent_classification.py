"""Intent-classification facade for the patient RAG agent.

The concrete implementation lives in :mod:`backend.services.agent_intent_router`.
This module gives the split RAG architecture a responsibility-named public
surface without changing behavior.
"""

from backend.services.agent_intent_router import route_intent  # noqa: F401

__all__ = ["route_intent"]
