"""Fetch and summarize news using RSS + Claude."""

from __future__ import annotations

import json
from datetime import datetime

import anthropic
import feedparser
import httpx

# Curated RSS feeds
FEEDS = {
    "Tech": [
        "https://news.ycombinator.com/rss",
        "https://feeds.arstechnica.com/arstechnica/technology-lab",
    ],
    "World": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    ],
    "Science": [
        "https://www.nature.com/nature.rss",
        "https://feeds.arstechnica.com/arstechnica/science",
    ],
    "AI": [
        "https://feeds.feedburner.com/TheHackersNews",
    ],
}

SUMMARIZE_PROMPT = """\
You are a news editor preparing a visual briefing for a large display.
Given these raw RSS entries, select the 6-8 most interesting/important stories
and produce a JSON object for rendering.

Rules:
- Headlines should be concise and punchy (max 12 words)
- Summaries should be 1-2 sentences, informative, in the same language as the original
- Distribute stories across 3 columns (2-3 stories each)
- Mark the single most important story as "featured": true
- For each story include: category, headline, summary, source (publication name)

Raw entries:
{entries}

Respond with ONLY valid JSON matching this schema:
{{
  "title": "MORNING BRIEF",
  "date": "{date}",
  "update_time": "{time}",
  "weather": "",
  "temp": "",
  "columns": [
    [
      {{"category": "TECH", "headline": "...", "summary": "...", "source": "Ars Technica", "featured": false}},
      ...
    ],
    [...],
    [...]
  ]
}}
"""


async def fetch_feeds() -> list[dict]:
    """Fetch all RSS feeds and return parsed entries."""
    entries = []
    async with httpx.AsyncClient(timeout=15) as client:
        for category, urls in FEEDS.items():
            for url in urls:
                try:
                    resp = await client.get(url)
                    feed = feedparser.parse(resp.text)
                    for entry in feed.entries[:5]:  # top 5 per feed
                        entries.append({
                            "category": category,
                            "title": entry.get("title", ""),
                            "summary": entry.get("summary", "")[:300],
                            "link": entry.get("link", ""),
                            "source": feed.feed.get("title", url),
                        })
                except Exception as e:
                    print(f"  Warning: failed to fetch {url}: {e}")
    return entries


async def curate_news() -> dict:
    """Fetch feeds and use Claude to curate into poster format."""
    print("Fetching RSS feeds...")
    entries = await fetch_feeds()
    print(f"  Got {len(entries)} entries from {len(FEEDS)} categories")

    entries_text = "\n\n".join(
        f"[{e['category']}] {e['title']}\n{e['summary']}\nSource: {e['source']}"
        for e in entries
    )

    now = datetime.now()
    prompt = SUMMARIZE_PROMPT.format(
        entries=entries_text,
        date=now.strftime("%A, %B %d, %Y"),
        time=now.strftime("%H:%M"),
    )

    print("Asking Claude to curate...")
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    return json.loads(raw)


def curate_news_sync() -> dict:
    import asyncio
    return asyncio.run(curate_news())
