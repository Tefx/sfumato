"""Painting-news matching: LLM-based semantic selection.

Selects the best painting for the current news tone by asking the LLM
to compare painting descriptions with news tone. No embedding needed.
"""

from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING

from sfumato.llm import LlmError, invoke_text, parse_json_response

if TYPE_CHECKING:
    from sfumato.config import AiConfig
    from sfumato.paintings import PaintingInfo

__all__ = [
    "MatcherError",
    "select_painting",
]


class MatcherError(Exception):
    """Raised when painting selection cannot proceed."""

    pass


MATCH_PROMPT = """\
You are selecting a painting to display alongside a news briefing on a TV.

NEWS TONE:
{tone}

AVAILABLE PAINTINGS:
{paintings}

Select the painting whose mood, atmosphere, and themes best complement the news tone.
Consider emotional resonance, not literal subject matter.

Output ONLY a JSON object:
{{"selected_index": 0, "reason": "brief explanation"}}

The index is zero-based into the paintings list above.
"""


async def select_painting(
    news_tone: str,
    paintings: list["PaintingInfo"],
    painting_descriptions: dict[str, str],
    ai_config: "AiConfig",
    strategy: str = "semantic",
    **kwargs,  # Accept and ignore legacy embedding_cache param
) -> tuple["PaintingInfo", float]:
    """Select the best painting for the given news tone.

    Uses LLM to directly compare painting descriptions with news tone.
    No embedding computation needed.

    Args:
        news_tone: Free-form tone description from news curation.
        paintings: List of available paintings to choose from.
        painting_descriptions: Mapping from content_hash to painting description.
        ai_config: AI backend configuration for LLM invocation.
        strategy: "semantic" (LLM matching) or "random".

    Returns:
        Tuple of (selected_painting, match_score).
        score is 1.0 for semantic match, 0.0 for random.
    """
    if not paintings:
        raise MatcherError("Cannot select painting: paintings list is empty")

    if strategy == "random":
        return (random.choice(paintings), 0.0)

    # Filter to paintings that have descriptions
    candidates = [
        (p, painting_descriptions.get(p.content_hash, ""))
        for p in paintings
        if p.content_hash in painting_descriptions and painting_descriptions[p.content_hash]
    ]

    if not candidates:
        # No descriptions available, fall back to random
        return (random.choice(paintings), 0.0)

    # Build paintings list for prompt
    paintings_text = "\n".join(
        f"[{i}] {p.title} by {p.artist}: {desc[:200]}"
        for i, (p, desc) in enumerate(candidates)
    )

    prompt = MATCH_PROMPT.format(tone=news_tone, paintings=paintings_text)

    try:
        response = await invoke_text(
            prompt=prompt,
            ai_config=ai_config,
            max_tokens=200,
            timeout_seconds=30,
        )
        data = parse_json_response(response.text)
        selected_index = int(data.get("selected_index", 0))

        if 0 <= selected_index < len(candidates):
            return (candidates[selected_index][0], 1.0)
        else:
            return (candidates[0][0], 1.0)

    except (LlmError, ValueError, KeyError, TypeError) as e:
        # LLM matching failed, fall back to random
        return (random.choice(paintings), 0.0)
