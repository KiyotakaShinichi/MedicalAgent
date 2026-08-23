"""The claim-boundary invariant every governance artifact must satisfy.

Each artifact's `claim_boundary` has to contain this phrase verbatim. It lives
here rather than in one of the four domain modules because all four are held to
it, and the tests assert it against each of them.
"""

from __future__ import annotations

REQUIRED_CLAIM_BOUNDARY_PHRASE = "not clinical validation"

__all__ = ["REQUIRED_CLAIM_BOUNDARY_PHRASE"]
