"""Render news data into a 4K poster image for The Frame."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from playwright.async_api import async_playwright

TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def _build_story_html(story: dict) -> str:
    cls = "story featured" if story.get("featured") else "story"
    parts = [f'<div class="{cls}">']
    if cat := story.get("category"):
        parts.append(f'  <div class="category">{cat}</div>')
    parts.append(f'  <div class="headline">{story["headline"]}</div>')
    if summary := story.get("summary"):
        parts.append(f'  <div class="summary">{summary}</div>')
    if source := story.get("source"):
        parts.append(f'  <div class="source">{source}</div>')
    parts.append("</div>")
    return "\n".join(parts)


def build_html(news: dict) -> str:
    """Build the full HTML from news data.

    news = {
        "title": "MORNING BRIEF",
        "date": "2026-03-14",
        "update_time": "08:00",
        "weather": "Partly Cloudy",
        "temp": "18°C",
        "columns": [
            [{"category": "TECH", "headline": "...", "summary": "...", "source": "...", "featured": True}, ...],
            [...],
            [...]
        ]
    }
    """
    template = (TEMPLATE_DIR / "news_poster.html").read_text()

    columns_html = []
    for col_stories in news.get("columns", []):
        stories_html = '\n<div class="divider"></div>\n'.join(
            _build_story_html(s) for s in col_stories
        )
        columns_html.append(f'<div class="column">\n{stories_html}\n</div>')

    html = template.replace("{{TITLE}}", news.get("title", "DAILY BRIEF"))
    html = html.replace("{{DATE}}", news.get("date", datetime.now().strftime("%Y-%m-%d")))
    html = html.replace("{{UPDATE_TIME}}", news.get("update_time", datetime.now().strftime("%H:%M")))
    html = html.replace("{{WEATHER}}", news.get("weather", ""))
    html = html.replace("{{TEMP}}", news.get("temp", ""))
    html = html.replace("{{COLUMNS}}", "\n".join(columns_html))

    return html


async def render_poster(news: dict, output_path: Path | None = None) -> Path:
    """Render news dict to a 4K PNG image."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"poster_{ts}.png"

    html = build_html(news)

    # Save HTML for debugging
    html_path = output_path.with_suffix(".html")
    html_path.write_text(html)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 3840, "height": 2160})
        await page.set_content(html, wait_until="networkidle")
        await page.screenshot(path=str(output_path), type="png")
        await browser.close()

    return output_path


def render_poster_sync(news: dict, output_path: Path | None = None) -> Path:
    return asyncio.run(render_poster(news, output_path))
