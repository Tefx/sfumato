"""Prototype script — run this to validate the full pipeline."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from . import render, news, tv

SAMPLE_NEWS = {
    "title": "MORNING BRIEF",
    "date": "Saturday, March 14, 2026",
    "update_time": "08:00",
    "weather": "Partly Cloudy",
    "temp": "18°C",
    "columns": [
        [
            {
                "category": "AI",
                "headline": "Claude 4.6 Opus Sets New Benchmark in Reasoning",
                "summary": "Anthropic's latest model achieves state-of-the-art on complex multi-step tasks, with significant improvements in code generation and mathematical reasoning.",
                "source": "Ars Technica",
                "featured": True,
            },
            {
                "category": "TECH",
                "headline": "Apple Unveils M5 Ultra at Spring Event",
                "summary": "The new chip targets AI workloads with 80-core Neural Engine and unified memory up to 512GB.",
                "source": "The Verge",
            },
        ],
        [
            {
                "category": "SCIENCE",
                "headline": "JWST Captures New Image of Exoplanet Atmosphere",
                "summary": "The telescope's spectroscopic data reveals complex organic molecules in a super-Earth's atmosphere, intensifying the search for biosignatures.",
                "source": "Nature",
            },
            {
                "category": "WORLD",
                "headline": "EU Passes Landmark Digital Infrastructure Act",
                "summary": "New regulations require cloud providers to offer data portability and interoperability across all member states.",
                "source": "BBC News",
            },
        ],
        [
            {
                "category": "BUSINESS",
                "headline": "Global Chip Demand Surges on AI Datacenter Boom",
                "summary": "TSMC reports record quarterly revenue as hyperscalers race to build inference capacity.",
                "source": "Financial Times",
            },
            {
                "category": "OPEN SOURCE",
                "headline": "Linux 7.0 Kernel Released with Rust Driver Support",
                "summary": "Major milestone as Rust becomes a first-class language for kernel module development.",
                "source": "Hacker News",
            },
        ],
    ],
}


async def run_step1_render_only():
    """Step 1: Verify we can generate a good-looking poster."""
    print("=" * 60)
    print("STEP 1: Render poster from sample data")
    print("=" * 60)

    path = await render.render_poster(SAMPLE_NEWS)
    print(f"\n✓ Poster rendered: {path}")
    print(f"  HTML:  {path.with_suffix('.html')}")
    print(f"  Open in browser to preview:")
    print(f"    open {path.with_suffix('.html')}")
    print(f"  Or view the PNG:")
    print(f"    open {path}")
    return path


async def run_step2_live_news():
    """Step 2: Fetch real news and render."""
    print("\n" + "=" * 60)
    print("STEP 2: Fetch live news + Claude curation + render")
    print("=" * 60)

    news_data = await news.curate_news()
    print(f"\n  Curated {sum(len(c) for c in news_data.get('columns', []))} stories")

    path = await render.render_poster(news_data)
    print(f"\n✓ Live poster rendered: {path}")
    print(f"    open {path}")
    return path


async def run_step3_tv_test(tv_ip: str):
    """Step 3: Test TV connection."""
    print("\n" + "=" * 60)
    print("STEP 3: Test Samsung The Frame connection")
    print("=" * 60)

    results = tv.test_connection(tv_ip)
    print(f"\nResults: {json.dumps(results, indent=2)}")
    return results


async def main():
    args = sys.argv[1:]

    if not args or "render" in args:
        await run_step1_render_only()

    if "live" in args:
        await run_step2_live_news()

    if "tv" in args:
        tv_ip = args[args.index("tv") + 1] if len(args) > args.index("tv") + 1 else None
        if not tv_ip:
            print("Usage: python -m frame_terminal.proto tv <TV_IP>")
        else:
            await run_step3_tv_test(tv_ip)


if __name__ == "__main__":
    asyncio.run(main())
