"""Global pytest configuration and fixtures for TB Screening test suite."""

import sys
import pytest

# Configure Windows stream encoding to UTF-8 to prevent charmap/cp1252 UnicodeEncodeError
# when handling or printing Devanagari (Hindi/Marathi) strings.
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


@pytest.fixture(autouse=True)
def force_offline_intake(monkeypatch, request):
    """Force deterministic offline rule-based fallback for all test cases by default.
    
    If GEMINI_API_KEY is present in the developer's environment or .env, unit tests
    must not unexpectedly make non-deterministic, billable external LLM calls.
    Tests specifically targeting live API (e.g. test_live_gemini_llm_intake) are exempted.
    """
    if "test_live_gemini" in request.node.name:
        return
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

