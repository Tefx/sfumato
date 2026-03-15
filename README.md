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

Every 5 minutes, sfumato selects a painting that matches the current news mood, analyzes its composition via LLM vision, pulls the next batch of curated news from the queue, renders a 4K image with text placed in the painting's quiet zones, and pushes it to your Frame TV.

## Features

- **Full-screen paintings** — The Frame's bezel is the frame. The painting fills the entire screen.
- **LLM-driven layout** — Each painting is analyzed by an LLM to find optimal text placement. Bright areas get dark text, dark areas get light text. The LLM recommends how many stories fit (2-7).
- **Semantic art–news matching** — An LLM compares painting descriptions with news tone to find the best emotional match. No fixed categories.
- **QR codes** — Each news story gets a scannable QR code (grouped in a corner with frosted glass background). Scan with your phone to read the full article. Configurable size and toggle.
- **Whisper text** — Art facts blend into the painting's quiet zones: small text carrying historical trivia or artist context. One fact per rotation.
- **News replay** — Consumed news batches replay until expired. The TV always has content to show.
- **Subject avoidance** — The LLM identifies the primary subject in each painting, ensuring text never obscures the focal point.
- **Smart news curation** — All RSS entries are kept (only obvious spam filtered). LLM translates and summarizes in batches of 15.
- **Configurable language** — Output in any language: Chinese, English, Japanese, etc.
- **Always-on display** — Inside active hours: news + paintings. Outside active hours: pure artwork. Never idle.
- **Cloud art sources** — Paintings from Met Museum and Wikimedia Commons. Locally cached, never repeats until pool exhausted.
- **Container-ready** — Runs as a `watch` daemon. Deploy on Synology NAS, Mac Mini, or any Docker host.

## Quick Start

```bash
# Install
pip install -e .
playwright install chromium

# Initialize (interactive setup + fetch seed paintings)
sfumato init

# Single run (for testing)
sfumato run                     # Pick art → analyze → fetch news → render → upload
sfumato run --no-upload         # Render only
sfumato run --painting my.jpg   # Use a specific painting

# Daemon mode (recommended)
sfumato watch -v                # Long-running, rotates every 5 minutes

# TV management
sfumato tv status               # Check TV connection and Art Mode
sfumato tv list                 # List uploaded images
sfumato tv clean                # Remove old uploads
```

## Configuration

```toml
# ~/.config/sfumato/config.toml

[tv]
ip = "192.168.88.22"
port = 8002
max_uploads = 5

[schedule]
news_interval_hours = 6         # How often to fetch + curate news
rotate_interval_minutes = 5     # How often to switch painting + news batch
active_hours = [10, 2]          # 10am-2am: news+art. 2am-10am: pure art.

[news]
language = "zh"
max_age_days = 3
expire_days = 7
replay_expire_days = 2
qr_size = 60                    # QR code size in pixels (0 to disable)
qr_enabled = true

[[news.feeds]]
name = "TLDR Tech"
url = "https://tldr.tech/api/rss/tech"
category = "Tech"

# ... add as many feeds as you want

[paintings]
cache_dir = "~/.sfumato/paintings"
seed_size = 50
pool_size = 200
sources = ["met", "wikimedia"]
match_strategy = "semantic"

[ai]
backend = "sdk"                 # sdk | cli
sdk_provider = "openrouter"     # openrouter | google | openai
model = "gemini-3-flash-preview"
# api_key = "sk-or-..."        # Or set OPENROUTER_API_KEY env var
```

## Docker Deployment

```bash
# 1. Create data directory
mkdir -p data/{paintings,state,output}

# 2. Create config file (see Configuration above)
cp config.example.toml data/config.toml
# Edit data/config.toml with your TV IP, API key, feeds

# 3. Build and start
export OPENROUTER_API_KEY="your-key"
docker compose up -d

# 4. Check logs
docker compose logs -f sfumato
```

### Schedule

```toml
[schedule]
# active_hours: inside = news+art, outside = pure art (no news)
# The TV always shows something.
# Examples:
#   active_hours = [10, 2]    # 10am to 1:59am (wraps midnight)
#   active_hours = [7, 23]    # 7am to 10:59pm
#   active_hours = [0, 24]    # news always on
```

### Volumes

| Mount | Purpose |
|-------|---------|
| `data/config.toml` → `/data/config.toml` | Configuration |
| `data/paintings/` → `/data/paintings/` | Art cache |
| `data/state/` → `/data/state/` | News queue, layout cache |
| `data/output/` → `/data/output/` | Rendered PNGs (debug) |

### Management

```bash
docker compose logs --tail 20 sfumato   # View status
docker compose restart sfumato          # Restart after config change
docker compose stop sfumato             # Graceful stop (90s grace)
git pull && docker compose up -d --build # Update
```

## Requirements

- Python 3.12+
- Chromium (via Playwright)
- Samsung The Frame TV (2020+, same network)
- OpenRouter API key (or Google/OpenAI API key)

## License

MIT
