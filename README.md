# sfumato

Turn your Samsung The Frame TV into a living art + news terminal.

Sfumato displays famous paintings full-screen on The Frame, with curated news text softly blended into the painting's natural quiet areas — sky, fog, water, shadows. An LLM analyzes each painting's composition to find the optimal text placement, colors, and density, so the information feels like part of the artwork, not overlaid on top of it.

Named after Leonardo da Vinci's *sfumato* technique — the smoky, borderless blending of tones — because that's exactly how text meets canvas here.

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐     ┌──────────┐
│  Art Sources │────▶│ LLM Analysis │────▶│  Renderer  │────▶│ Frame TV │
│  (cloud)     │     │ (composition)│     │ (4K PNG)   │     │ (upload) │
└─────────────┘     └──────────────┘     └────────────┘     └──────────┘
                           ▲                    ▲
                    ┌──────┘              ┌─────┘
               ┌────────────┐      ┌───────────┐
               │ RSS Feeds  │─────▶│ LLM News  │
               │ (multiple) │      │ (curate,  │
               └────────────┘      │  translate)│
                                   └───────────┘
```

Every 15 minutes, sfumato selects a painting that matches the current news mood, analyzes its composition via LLM vision, pulls the next batch of curated news from the queue, renders a 4K image with text placed in the painting's quiet zones, and pushes it to your Frame TV.

## Features

- **Full-screen paintings** — The Frame's bezel is the frame. No borders, no panels, no picture-in-picture. The painting fills the screen.
- **LLM-driven layout** — Each painting is analyzed by an LLM (default: OpenRouter + Gemini 3 Flash via SDK, or Gemini/Codex/Claude CLI) to find optimal text placement. Bright areas get dark text, dark areas get light text. The LLM recommends how many stories fit (2-7).
- **Semantic art–news matching** — No fixed mood categories. Each painting gets a free-form LLM description of its emotional tone, themes, and atmosphere. Each news batch gets the same. An LLM directly compares painting descriptions with news tone to find the best match. "Stormy skies with golden fields" naturally matches "industry upheaval amid golden-age AI breakthroughs."
- **Whisper text** — Art facts blend into the painting's quiet zones: small, unobtrusive text carrying historical trivia, artist context, or compositional insights. Each painting displays one fact per rotation, cycling through 1-3 facts for repeated viewings.
- **News replay** — Previously seen news batches cycle through on subsequent rotations when the primary queue empties. News facts are replayed with a configurable expiration window, keeping your backlog fresh without redundant alerts.
- **Subject avoidance** — The LLM identifies the primary subject zone in each painting, ensuring news overlays never obscure the focal point. The subject region is preserved as visually clean space.
- **Smart news curation** — Multiple RSS sources are fetched and deduplicated by URL, then an LLM filters obvious spam, ranks, summarizes, and translates to your configured language in batches of 15. All non-spam stories are kept. Complete stories, not just headlines. Seen article URLs are tracked to prevent repetition across refresh cycles.
- **Time-window awareness** — Fetches news from the past N days (default 3), not just the latest. If you've been away, you'll catch up on what matters. Articles older than 7 days are expired.
- **Configurable display language** — Output in any language: Chinese, English, Japanese, etc. The LLM translates and adapts summaries accordingly.
- **Dual refresh cycle** — News is fetched every 6 hours (configurable), paintings rotate every 15 minutes. News is queued in batches of 8 stories; each rotation renders the count determined by the painting's layout analysis (2-7 stories). Consumed batches replay until expired.
- **Cloud art sources** — Paintings from Met Museum and Wikimedia Commons APIs. Locally cached, never repeats until the full pool is exhausted.
- **Layout caching** — Each painting's composition analysis and description are cached by content hash. One LLM call per painting, forever.
- **Seed art library** — `sfumato init` pre-fetches 50 paintings from cloud APIs covering diverse styles and moods, then analyzes them all. The daemon continues backfilling to 200+ in the background.
- **TV-aware** — Detects if the TV is off or not in Art Mode before pushing. Auto-cleans old uploads.
- **Always-on display** — The TV always shows something. Inside active hours: paintings with news overlay. Outside active hours: pure artwork, no news.
- **Multi-template** — Landscape paintings with whitespace get text blended in. Portrait paintings get a side-panel layout. Dense compositions get a magazine split.
- **Container-ready** — Runs as a long-lived `watch` daemon. Deploy on Synology NAS, Mac Mini, or any Docker host.

## Quick Start

```bash
# Install
pip install sfumato
playwright install chromium

# Initialize (creates config + fetches 50 seed paintings + analyzes them)
sfumato init

# Single run (for testing)
sfumato run                     # Pick art → analyze → fetch news → render → upload
sfumato run --no-upload         # Render only, open preview
sfumato run --painting my.jpg   # Use a specific painting

# Daemon mode
sfumato watch                   # Long-running, manages all timers

# TV management
sfumato tv status               # Check TV connection and Art Mode
sfumato tv list                 # List uploaded images
sfumato tv clean                # Remove old uploads

# Preview
sfumato preview                 # Render and open in system viewer
```

## Configuration

```toml
# ~/.config/sfumato/config.toml

[tv]
ip = "192.168.88.22"
port = 8002
max_uploads = 5                     # Auto-clean older uploads

[schedule]
news_interval_hours = 6             # How often to fetch + curate news
rotate_interval_minutes = 15        # How often to switch painting + news batch
active_hours = [10, 2]              # Inside: news + art. Outside: pure art, no news

[news]
language = "zh"                     # Display language (zh, en, ja, ...)
                                    # All stories kept (only spam filtered), processed in batches of 15
max_age_days = 3                    # Fetch articles up to N days old
expire_days = 7                     # Discard articles older than this
replay_expire_days = 2              # Expire replay batches older than this

[[news.feeds]]
name = "TLDR Tech"
url = "https://tldr.tech/api/rss/tech"
category = "Tech"

[[news.feeds]]
name = "TLDR AI"
url = "https://tldr.tech/api/rss/ai"
category = "AI"

[[news.feeds]]
name = "Hacker News 100+"
url = "https://hnrss.org/frontpage?points=100"
category = "Tech"

[[news.feeds]]
name = "Ars Technica"
url = "https://feeds.arstechnica.com/arstechnica/index"
category = "Tech"

[[news.feeds]]
name = "TechCrunch"
url = "https://techcrunch.com/feed/"
category = "Tech"

[[news.feeds]]
name = "The Verge"
url = "https://www.theverge.com/rss/index.xml"
category = "Tech"

[[news.feeds]]
name = "Import AI"
url = "https://importai.substack.com/feed"
category = "AI"

[[news.feeds]]
name = "Latent Space"
url = "https://www.latent.space/feed"
category = "AI"

[[news.feeds]]
name = "BBC News"
url = "https://feeds.bbci.co.uk/news/rss.xml"
category = "World"

[[news.feeds]]
name = "Reuters"
url = "http://feeds.reuters.com/reuters/topNews"
category = "World"

[[news.feeds]]
name = "NYT Home"
url = "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"
category = "World"

[[news.feeds]]
name = "Nature"
url = "https://www.nature.com/nature.rss"
category = "Science"

[[news.feeds]]
name = "MIT Tech Review"
url = "https://www.technologyreview.com/feed/"
category = "Science"

[[news.feeds]]
name = "Ars Technica Science"
url = "https://feeds.arstechnica.com/arstechnica/science"
category = "Science"

[[news.feeds]]
name = "Financial Times"
url = "https://www.ft.com/rss/home"
category = "Economy"

[paintings]
cache_dir = "~/.sfumato/paintings"
seed_size = 50                      # Pre-fetch during `sfumato init`
pool_size = 200                     # Background backfill target
sources = ["met", "wikimedia"]
match_strategy = "semantic"         # semantic | random

[ai]
backend = "sdk"                     # sdk | cli
sdk_provider = "openrouter"         # openrouter | google | openai (when backend="sdk")
model = "gemini-3-flash-preview"    # Model for layout analysis and news curation
```

## How Layout Analysis Works

For each new painting, sfumato calls the configured LLM with the image and asks it to:

1. **Identify quiet zones** — Sky, fog, water, shadows, flat backgrounds
2. **Choose text placement** — CSS position for the text zone
3. **Set colors** — Adaptive text color based on zone brightness
4. **Recommend density** — How many stories fit without visual noise
5. **Design scrim** — A subtle gradient to ensure readability without visible overlay
6. **Tag mood and themes** — Emotional tone and subject matter for news matching

The result is cached by painting content hash — the same painting never needs re-analysis.

```
Bright painting (e.g., Monet)     Dark painting (e.g., Van Gogh)
┌─────────────────────────┐       ┌─────────────────────────┐
│              dark text ◄─┤       │                         │
│              on bright   │       │  light text ►───────────┤
│              sky area    │       │  on dark                │
│                          │       │  lower area             │
│   ████ painting ████     │       │   ████ painting ████    │
│   ████ subject  ████     │       │   ████ subject  ████    │
└─────────────────────────┘       └─────────────────────────┘
```

## Semantic Art–News Matching

No predefined mood categories. Instead, sfumato uses free-form descriptions and LLM-based matching:

```
Painting analysis (one-time, cached):
  LLM sees painting → generates rich description:
  "暴风雨前的宁静，灰蓝色天空压迫着金色麦田，
   孤独感与自然的壮美交织，带有不安的预兆感"
  → store description

News batch (each rotation):
  LLM reads stories → describes overall tone:
  "科技巨头间的紧张博弈，收购与反垄断交织，
   行业格局快速重组，充满不确定性"
  → LLM compares tone with all painting descriptions
  → pick best match
```

This captures nuances that fixed labels cannot: the difference between "quiet melancholy" and "peaceful solitude," or between "anxious energy" and "joyful momentum." The matching is emergent, not hand-coded.

Painting descriptions are cached forever (one LLM call per painting). News batch descriptions are generated during curation (no extra LLM call -- part of the same prompt).

## Requirements

- Python 3.12+
- Chromium (installed via Playwright)
- Samsung The Frame TV (2020+ models, on same network)
- Default: OpenRouter API key (`OPENROUTER_API_KEY`) for SDK backend. Alternatively, one of: `gemini` CLI, `codex` CLI, or `claude` CLI for CLI backend mode

## Docker Deployment

### Quick Start (Synology NAS / Mac Mini / any Docker host)

```bash
# 1. Create data directory
mkdir -p data/{paintings,state,output}

# 2. Create config file
cat > data/config.toml << 'EOF'
data_dir = "/data"

[tv]
ip = "192.168.88.22"       # Your Samsung The Frame TV IP
port = 8002

[schedule]
news_interval_hours = 6     # How often to fetch new RSS
rotate_interval_minutes = 15 # How often to switch painting + news
active_hours = [10, 2]      # Inside: news + art. Outside: pure art, no news

[news]
language = "zh"             # Display language (zh/en/ja/...)

[[news.feeds]]
name = "TLDR Tech"
url = "https://tldr.tech/api/rss/tech"
category = "Tech"

[[news.feeds]]
name = "BBC News"
url = "https://feeds.bbci.co.uk/news/rss.xml"
category = "World"

# Add more feeds as needed — all entries are kept (only obvious spam filtered)

[paintings]
cache_dir = "/data/paintings"
sources = ["met", "wikimedia"]

[ai]
backend = "sdk"
sdk_provider = "openrouter"
model = "gemini-3-flash-preview"
EOF

# 3. Set API key
export OPENROUTER_API_KEY="your-key-here"

# 4. Build and start
docker compose up -d

# 5. Check logs
docker compose logs -f sfumato
```

### Configuration Reference

```toml
[schedule]
# active_hours: inside this window, news + art. Outside, pure art (no news).
# The TV always shows something -- never idle.
# Examples:
#   active_hours = [10, 2]    # 10am to 1:59am next day (wraps midnight)
#   active_hours = [7, 23]    # 7am to 10:59pm
#   active_hours = [10, 14]   # 10am to 1:59pm only
#   active_hours = [0, 24]    # all day (news always on)
```

### Volumes

| Mount | Purpose |
|-------|---------|
| `data/config.toml` → `/data/config.toml` | Your configuration |
| `data/paintings/` → `/data/paintings/` | Downloaded art cache (persists across restarts) |
| `data/state/` → `/data/state/` | News queue, replay queue, layout cache |
| `data/output/` → `/data/output/` | Rendered PNGs (optional, for debugging) |

### Management

```bash
# View status
docker compose logs --tail 20 sfumato

# Restart (e.g. after config change)
docker compose restart sfumato

# Stop gracefully (waits up to 90s for current action to finish)
docker compose stop sfumato

# Update to latest
git pull && docker compose build && docker compose up -d
```

### Network

The container uses `network_mode: host` so it can reach your TV on the local network. On macOS Docker Desktop, you may need to use the TV's IP directly (host networking works differently).

## License

MIT
