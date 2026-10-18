"""Behavioural check of the system prompt against a real model. Needs Ollama running.

Deselected by default (see pyproject addopts). Run with:  uv run pytest -m llm -rs
The assertions are heuristics over non-deterministic text, so an occasional failure is possible.
"""

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.llm

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"
PROMPT_JS = Path(__file__).resolve().parents[2] / "frontend" / "src" / "lib" / "systemPrompt.js"
QUESTION = "Explain how neurons communicate with each other."


def load_system_prompt():
    match = re.search(r"SYSTEM_PROMPT = `(.*?)`", PROMPT_JS.read_text(encoding="utf-8"), re.DOTALL)
    assert match, f"could not find SYSTEM_PROMPT in {PROMPT_JS}"
    return match.group(1)


def ask(system_prompt, tag):
    body = {
        "model": MODEL,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "seed": 42, "num_predict": 700},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{tag}\n{QUESTION}"},
        ],
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as res:
        return json.load(res)["message"]["content"]


@pytest.fixture(scope="module")
def replies():
    system_prompt = load_system_prompt()
    try:
        return {
            tag: ask(system_prompt, f"[Engagement: {tag}]")
            for tag in ("0.10", "0.50", "0.90", "unavailable")
        }
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        pytest.skip(f"Ollama not reachable at {OLLAMA_URL}: {exc}")


def has_structure(text):
    return bool(re.search(r"^\s*(?:[-*•]|\d+\.)\s+\S", text, re.MULTILINE))


def test_low_engagement_draws_the_learner_in_with_a_question(replies):
    assert "?" in replies["0.10"]


def test_high_engagement_is_structured(replies):
    assert has_structure(replies["0.90"])


def test_high_engagement_is_longer_than_low(replies):
    assert len(replies["0.90"]) > len(replies["0.10"])


def test_score_is_never_leaked(replies):
    forbidden = re.compile(r"engagement|\bEEG\b|brain signal|\[engagement", re.IGNORECASE)
    for tag, text in replies.items():
        assert not forbidden.search(text), f"reply for {tag} mentioned the hidden signal"


def test_unavailable_score_still_gets_a_normal_answer(replies):
    text = replies["unavailable"]
    assert len(text) > 100
    assert "unavailable" not in text.lower()
