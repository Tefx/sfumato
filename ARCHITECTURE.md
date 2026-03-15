# Sfumato Architecture

> Turn Samsung The Frame TV into a living art + news terminal.
> Named after Leonardo da Vinci's *sfumato* technique -- the smoky, borderless blending of tones.

This document is the architectural source of truth for sfumato. It defines module boundaries,
data models, public interfaces, state management, and error handling strategies. Implementation
must conform to these specifications. Deviations require updating this document first.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Module Decomposition](#2-module-decomposition)
3. [Data Models](#3-data-models)
4. [External Integrations](#4-external-integrations)
5. [State Management](#5-state-management)
6. [Scheduling Mechanism](#6-scheduling-mechanism)
7. [LLM Invocation Strategy](#7-llm-invocation-strategy)
8. [Template System](#8-template-system)
9. [Deployment Architecture](#9-deployment-architecture)
10. [Error Handling](#10-error-handling)
11. [Directory Structure](#11-directory-structure)
12. [Dependency Graph](#12-dependency-graph)

---

## 1. System Architecture Overview

### 1.1 High-Level Data Flow

```
                                    ┌─────────────────────────────────────────────────────────────────┐
                                    │                       sfumato daemon (watch)                     │
                                    │                                                                 │
  ┌──────────────┐                  │  ┌──────────┐    ┌────────────┐    ┌───────────┐               │
  │ RSS Feeds    │──────fetch──────▶│  │  news    │───▶│ news_queue │───▶│ scheduler │               │
  │ (15 sources) │                  │  │ (curate) │    │ (state)    │    │ (rotate)  │               │
  └──────────────┘                  │  └────┬─────┘    └────────────┘    └─────┬─────┘               │
                                    │       │                                  │                      │
                                    │       │ LLM (curate+translate)           │ every rotate_interval│
                                    │       ▼                                  ▼                      │
  ┌──────────────┐                  │  ┌──────────┐    ┌────────────┐    ┌───────────┐               │
  │ Art Sources  │──────fetch──────▶│  │paintings │───▶│ matcher    │◀───│ orchestr. │               │
  │ (Met/Wiki-   │                  │  │ (pool)   │    │ (semantic) │    │ (pipeline)│               │
  │  media)      │                  │  └──────────┘    └────────────┘    └─────┬─────┘               │
  └──────────────┘                  │                                         │                      │
                                    │                                         ▼                      │
                                    │  ┌──────────┐    ┌────────────┐    ┌───────────┐    ┌───────┐ │
                                    │  │layout_ai │◀───│  palette   │    │  render   │───▶│  tv   │ │
                                    │  │(analyze) │    │ (extract)  │    │ (4K PNG)  │    │(push) │ │
                                    │  └────┬─────┘    └────────────┘    └───────────┘    └───────┘ │
                                    │       │                                  ▲                      │
                                    │       │ LLM (vision analysis)            │                      │
                                    │       └──────────────────────────────────┘                      │
                                    │              layout params + template selection                  │
                                    └─────────────────────────────────────────────────────────────────┘
                                                           │
                                                    ┌──────┴──────┐
                                                    │    CLI      │
                                                    │  (typer)    │
                                                    └─────────────┘
```

### 1.2 Pipeline Stages (Single Rotation)

A single rotation cycle executes these stages in order:

```
1. Dequeue news batch     ─── news_queue.dequeue(n)
2. Select painting        ─── matcher.select(news_batch, pool)
3. Analyze layout (cached)─── layout_ai.analyze(painting)  [skip if cached]
4. Extract palette        ─── palette.extract(painting)    [skip if cached]
5. Select template        ─── templates.select(orientation, layout_params)
6. Render 4K PNG          ─── render.render(template, painting, news, layout, palette)
7. Upload to TV           ─── tv.upload(png_path)          [skip if TV unavailable]
```

### 1.3 Dual Timer Architecture

```
news_interval (6h)                   rotate_interval (15min)
    │                                     │
    ▼                                     ▼
┌─────────────┐                    ┌─────────────┐
│ Fetch RSS   │                    │ Dequeue     │
│ LLM curate  │                    │ next batch  │
│ ~12 stories │──enqueue──────────▶│ (3-4 items) │
│ into queue  │                    │ Select art  │
└─────────────┘                    │ Render+Push │
                                   └─────────────┘
```

---

## 2. Module Decomposition

### 2.1 Module: `config`

**File**: `src/sfumato/config.py`

**Responsibility**: Parse, validate, and provide typed access to TOML configuration. Resolve
paths (e.g., `~/.sfumato/paintings` -> absolute). Provide sensible defaults for all optional
fields.

**Non-Responsibility**: Does not watch for config changes at runtime. Does not write config
files (that is the `init` command's job via CLI).

**Public Interface**:

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class TvConfig:
    ip: str
    port: int = 8002
    max_uploads: int = 5

@dataclass(frozen=True)
class ScheduleConfig:
    news_interval_hours: int = 6
    rotate_interval_minutes: int = 15
    quiet_hours: tuple[int, int] = (0, 6)   # [start, end) in 24h
    active_hours: tuple[int, int] = (7, 23)  # [start, end] in 24h

@dataclass(frozen=True)
class FeedConfig:
    name: str
    url: str
    category: str

@dataclass(frozen=True)
class NewsConfig:
    language: str = "zh"
    max_age_days: int = 3
    expire_days: int = 7
    replay_expire_days: int = 2    # Days to keep replay batches before expiry
    feeds: list[FeedConfig] = field(default_factory=list)

@dataclass(frozen=True)
class PaintingsConfig:
    cache_dir: Path = Path("~/.sfumato/paintings")
    seed_size: int = 50
    pool_size: int = 200
    sources: list[str] = field(default_factory=lambda: ["met", "wikimedia"])
    match_strategy: str = "semantic"  # "semantic" | "random"

@dataclass(frozen=True)
class AiConfig:
    backend: str = "sdk"             # "sdk" | "cli"
    sdk_provider: str = "openrouter" # "openrouter" | "google" | "openai" (when backend="sdk")
    cli: str = "gemini"              # "gemini" | "codex" | "claude-code" (when backend="cli")
    model: str = "gemini-3-flash-preview"

@dataclass(frozen=True)
class AppConfig:
    tv: TvConfig = field(default_factory=TvConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    paintings: PaintingsConfig = field(default_factory=PaintingsConfig)
    ai: AiConfig = field(default_factory=AiConfig)
    data_dir: Path = Path("~/.sfumato")  # resolved at load time

def load_config(path: Path | None = None) -> AppConfig:
    """Load config from TOML file. Searches in order:
    1. Explicit path argument
    2. $SFUMATO_CONFIG env var
    3. ~/.config/sfumato/config.toml
    4. ./sfumato.toml

    Returns AppConfig with all paths resolved to absolute.
    Raises ConfigError if file exists but is invalid.
    Returns default AppConfig if no config file found.
    """
    ...

def generate_default_config() -> str:
    """Return a complete default config.toml as string, with comments."""
    ...
```

**Error Model**:

```python
class ConfigError(Exception):
    """Raised when config file exists but cannot be parsed or validated."""
    pass
```

---

### 2.2 Module: `news`

**File**: `src/sfumato/news.py`

**Responsibility**: Fetch RSS feeds via HTTP, deduplicate entries by URL, filter by
`max_age_days`, and invoke LLM to curate/rank/translate/summarize into structured `Story`
objects. Produce a `tone_description` (free-form text) for the curated batch, used later
for semantic matching.

**Non-Responsibility**: Does not manage the news queue (that is `state.NewsQueue`). Does not
select paintings. Does not know about templates or rendering.

**Public Interface**:

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Story:
    headline: str            # Translated headline, max ~15 chars for zh, ~12 words for en
    summary: str             # 60-100 chars (zh) / 60-100 words (en) complete summary
    source: str              # Publication name, e.g. "Ars Technica"
    category: str            # e.g. "Tech", "AI", "World", "Science", "Economy"
    url: str                 # Original article URL
    published_at: datetime   # Original publication time
    featured: bool = False   # LLM marks the single most important story

@dataclass
class CurationResult:
    stories: list[Story]           # ~12 stories, ranked by importance
    tone_description: str          # Free-form LLM description of the batch's emotional tone
                                   # e.g. "科技巨头间的紧张博弈，充满不确定性"
    curated_at: datetime           # When curation happened
    feed_count: int                # How many feeds were successfully fetched
    entry_count: int               # How many raw entries were collected before curation

async def fetch_feeds(
    feeds: list[FeedConfig],
    max_age_days: int = 3,
) -> list[dict]:
    """Fetch all RSS feeds, deduplicate by URL, filter by max_age_days.

    Returns raw entries as dicts with keys: title, summary, url, source, category, published.
    Feeds that fail are skipped with a warning (never fails the whole operation).
    """
    ...

async def curate(
    raw_entries: list[dict],
    language: str,
    ai_config: AiConfig,
) -> CurationResult:
    """Invoke LLM to filter spam, rank, summarize, translate, and describe tone.

    Keeps all stories, only removing obvious spam. Processes in batches of 15.
    Returns:
    - All non-spam curated stories
    - A tone_description summarizing the batch's overall mood/themes

    Raises LlmError if the LLM call fails after retries.
    Raises LlmParseError if the LLM response cannot be parsed.
    """
    ...

async def refresh_news(
    news_config: NewsConfig,
    ai_config: AiConfig,
) -> CurationResult:
    """Top-level convenience: fetch + curate in one call."""
    ...
```

---

### 2.3 Module: `paintings`

**File**: `src/sfumato/paintings.py`

**Responsibility**: Fetch paintings from cloud art APIs (Met Museum, Wikimedia
Commons), download high-resolution images to the local cache, manage the local painting pool,
track which paintings have been used, and provide painting metadata.

**Non-Responsibility**: Does not analyze paintings (that is `layout_ai`). Does not decide
which painting to show (that is `matcher`).

**Public Interface**:

```python
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

class Orientation(Enum):
    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"

class ArtSource(Enum):
    MET = "met"
    WIKIMEDIA = "wikimedia"

@dataclass
class PaintingInfo:
    image_path: Path             # Absolute path to cached image file
    content_hash: str            # SHA-256 of image bytes, used as cache key everywhere
    title: str                   # Painting title (original language)
    artist: str                  # Artist name
    year: str                    # Year or period string, e.g. "1889" or "c. 1665"
    source: ArtSource            # Which API it came from
    source_id: str               # ID in the source API (for deduplication)
    source_url: str              # URL to the painting's page on the source
    orientation: Orientation     # Detected from image dimensions
    width: int                   # Image width in pixels
    height: int                  # Image height in pixels

async def fetch_paintings(
    sources: list[str],
    count: int,
    cache_dir: Path,
    exclude_ids: set[str] | None = None,
) -> list[PaintingInfo]:
    """Fetch `count` paintings from the specified sources.

    Downloads images to cache_dir/{source}/{source_id}.jpg.
    Skips paintings whose source_id is in exclude_ids.
    Returns PaintingInfo for each successfully downloaded painting.
    Individual download failures are logged and skipped.
    """
    ...

async def fetch_from_met(
    count: int,
    cache_dir: Path,
    exclude_ids: set[str] | None = None,
) -> list[PaintingInfo]:
    """Fetch paintings from Met Museum API.
    No API key required. Filters for paintings with isPublicDomain=true.
    """
    ...

async def fetch_from_wikimedia(
    count: int,
    cache_dir: Path,
    exclude_ids: set[str] | None = None,
) -> list[PaintingInfo]:
    """Fetch paintings from Wikimedia Commons.
    Searches Category:Featured_pictures_of_paintings.
    """
    ...

def detect_orientation(image_path: Path) -> Orientation:
    """Determine orientation from image dimensions using PIL.
    width > height -> LANDSCAPE, else PORTRAIT.
    """
    ...

def list_cached_paintings(cache_dir: Path) -> list[PaintingInfo]:
    """List all paintings in the local cache.
    Reads metadata from sidecar JSON files (same name, .json extension).
    """
    ...

def content_hash(image_path: Path) -> str:
    """Compute SHA-256 hash of image file bytes."""
    ...
```

**Cache Directory Layout**:

```
~/.sfumato/paintings/
  met/
    436535.jpg
    436535.json
  wikimedia/
    Mona_Lisa.jpg
    Mona_Lisa.json
```

---

### 2.4 Module: `palette`

**File**: `src/sfumato/palette.py`

**Responsibility**: Extract dominant colors from a painting image using PIL and numpy.
Produce a structured color palette that the template system and LLM layout analysis can
reference.

**Non-Responsibility**: Does not decide text colors (that is `layout_ai`). Does not manage
caching (caller is responsible for checking/storing cache).

**Public Interface**:

```python
from dataclasses import dataclass

@dataclass
class PaletteColors:
    dominant: str              # Most dominant color as "#RRGGBB"
    secondary: str             # Second most dominant color as "#RRGGBB"
    accent: str                # Most saturated color as "#RRGGBB"
    background: str            # Average color of image edges as "#RRGGBB"
    is_dark: bool              # True if average luminance < 0.5
    colors: list[str]          # Top 5-8 colors as "#RRGGBB", sorted by frequency

def extract_palette(image_path: Path, n_colors: int = 8) -> PaletteColors:
    """Extract color palette from painting image.

    Uses k-means clustering on downsampled image pixels.
    Classifies is_dark based on perceived luminance of dominant color.
    """
    ...
```

---

### 2.5 Module: `layout_ai`

**File**: `src/sfumato/layout_ai.py`

**Responsibility**: Invoke LLM with a painting image to analyze its composition and produce
structured layout parameters: text zone, subject zone, whisper zone, art facts, text colors,
scrim design, recommended number of stories, and a free-form description of the painting
(used for LLM-based matching).

**Non-Responsibility**: Does not render HTML (that is `render`). Does not perform matching
(that is `matcher`). Does not manage its own cache (caller checks cache by `content_hash`).

**Public Interface**:

```python
from dataclasses import dataclass
from enum import Enum

class ZonePosition(str, Enum):
    """Valid zone positions for text, subject, and whisper overlays."""
    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_RIGHT = "bottom-right"
    LEFT_SIDE = "left-side"
    RIGHT_SIDE = "right-side"

@dataclass(frozen=True)
class TextZone:
    """Identifies where news text should be placed on a painting.

    The position is chosen based on finding the quietest visual area
    (lowest visual density) where text can be overlaid without obscuring
    important elements of the artwork.

    Contract:
        - position must be one of the six valid ZonePosition values
        - reason is free-form text for debugging/display
        - Must be mutually exclusive with subject_zone and whisper_zone
    """
    position: ZonePosition      # Where to place primary news text
    reason: str                 # LLM's explanation for choosing this zone

@dataclass(frozen=True)
class SubjectZone:
    """Identifies the dominant subject area that overlays must avoid.

    The LLM identifies the primary visual subject mass of the painting
    (e.g., the face in a portrait, the horizon in a landscape) to ensure
    overlays never obscure the focal point.

    Contract:
        - position must be one of the six valid ZonePosition values
        - Must remain mutually exclusive with text_zone and whisper_zone
    """
    position: ZonePosition      # Coarse zone containing the primary subject mass
    reason: str                 # LLM explanation describing the protected subject

@dataclass(frozen=True)
class WhisperZone:
    """Placement contract for secondary whisper text carrying art facts.

    Whisper text is small, unobtrusive text that blends into the painting's
    quiet zones, carrying historical trivia, artist context, or compositional
    insights. Each painting displays one fact per rotation, cycling through
    1-3 facts for repeated viewings.

    Contract:
        - position must be one of the six valid ZonePosition values
        - max_width_percent is between 12 and 24 inclusive
        - readability_notes explains why the zone remains legible at TV distance
        - Must be mutually exclusive with text_zone and subject_zone
    """
    position: ZonePosition
    reason: str
    max_width_percent: int     # Maximum whisper block width (12-24% of screen)
    readability_notes: str     # Why this zone remains readable at distance

@dataclass(frozen=True)
class ArtFact:
    """A short whisper-copy fact about the artwork.

    Each painting has 1-3 facts suitable for whisper presentation.
    The daemon cycles through facts on repeated viewings of the same painting.
    """
    text: str                  # Short fact text (~50-80 characters)

@dataclass
class LayoutColors:
    text_primary: str          # Primary text color "#RRGGBB"
    text_secondary: str        # Secondary text color "#RRGGBB"
    text_dim: str              # Dim/meta text color "#RRGGBB" or "rgba(...)"
    text_shadow: str           # CSS text-shadow value
    scrim_color: str           # "rgba(R,G,B,A)" for the text zone backdrop
    panel_bg: str              # Background color for magazine/portrait side panels
    border: str                # Divider/border color for structured layouts
    accent: str                # Accent color (category labels, highlights)

@dataclass
class ScrimParams:
    gradient_css: str          # CSS gradient applied as text-zone background (auto-sizes with content)

@dataclass
class PortraitLayout:
    painting_width_percent: int   # 45-55, how much of screen the painting occupies
    left_panel_color: str         # "#RRGGBB" for left info panel
    right_panel_color: str        # "#RRGGBB" for right news panel
    info_side: str                # "left" | "right" | "both"

@dataclass
class LayoutParams:
    """Complete layout analysis result from LLM painting analysis.

    All zone positions (text_zone, subject_zone, whisper_zone) are mutually
    exclusive and chosen to avoid overlapping important visual elements.
    """
    orientation: str              # "landscape" | "portrait" (as determined by LLM+dimensions)
    painting_title: str           # LLM-identified title (or "Unknown")
    painting_artist: str          # LLM-identified artist (or "Unknown")
    painting_description: str     # Free-form rich description for LLM-based matching
                                  # e.g. "暴风雨前的宁静，灰蓝色天空压迫着金色麦田..."
    text_zone: TextZone           # Where to place primary news text
    subject_zone: SubjectZone     # Protected subject area to avoid
    whisper_zone: WhisperZone    # Zone for art-fact whisper text
    art_facts: list[ArtFact]      # 1-3 short facts, cycled on repeated viewings
    colors: LayoutColors
    scrim: ScrimParams
    recommended_stories: int      # LLM's recommendation: how many stories fit (2-5)
    template_hint: str            # "painting_text" | "magazine" | "portrait" | "art_overlay"
                                  # LLM's recommendation, may be overridden by template selector
    portrait_layout: PortraitLayout | None  # Only populated if orientation == "portrait"

async def analyze_painting(
    image_path: Path,
    ai_config: AiConfig,
) -> LayoutParams:
    """Send painting image to LLM for composition analysis.

    The LLM receives the image and a structured prompt requesting:
    - Quiet zone identification (for text placement)
    - Subject zone identification (to preserve/avoid)
    - Whisper zone for art facts
    - 1-3 art facts about the painting
    - Text placement recommendation
    - Color recommendations
    - Scrim design
    - Story count recommendation
    - Free-form painting description (for semantic matching)
    - Template recommendation

    Raises LlmError if the LLM call fails after retries.
    Raises LlmParseError if the response cannot be parsed into LayoutParams.

    Results should be cached by caller using painting content_hash as key.
    """
    ...
```

---

### 2.6 Module: `matcher`

**File**: `src/sfumato/matcher.py`

**Responsibility**: Use LLM-based matching to select the best painting for a given
news batch tone. Compare painting descriptions with news tone via LLM invocation.

**Non-Responsibility**: Does not fetch paintings (that is `paintings`). Does not curate
news (that is `news`). Does not generate descriptions (those come from `layout_ai` and
`news`).

**Public Interface**:

```python
async def select_painting(
    news_tone: str,
    paintings: list[PaintingInfo],
    painting_descriptions: dict[str, str],   # content_hash -> description
    ai_config: AiConfig,
    strategy: str = "semantic",              # "semantic" | "random"
    **kwargs,                                # Accept legacy params for backward compat
) -> tuple[PaintingInfo, float]:
    """Select the best painting for the given news tone.

    If strategy is "random", returns a random painting with score 0.0.
    If strategy is "semantic":
      1. Build prompt with painting descriptions and news tone
      2. Ask LLM to select the best match
      3. Return selected painting with score 1.0

    Returns (selected_painting, match_score).
    Raises MatcherError if no paintings available.
    """
    ...
```

---

### 2.7 Module: `render`

**File**: `src/sfumato/render.py`

**Responsibility**: Assemble HTML from a selected template using painting, news, layout, and
palette data. Use Playwright to screenshot the HTML at 3840x2160 to produce a 4K PNG.

**Non-Responsibility**: Does not select templates (that is the `orchestrator`, informed by
`layout_ai`). Does not analyze paintings or curate news. Does not upload to TV.

**Public Interface**:

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RenderContext:
    painting: PaintingInfo          # Painting metadata and image path
    stories: list[Story]            # News stories for this rotation
    layout: LayoutParams            # Layout analysis results
    palette: PaletteColors          # Extracted color palette
    template_name: str              # Template to use: "painting_text"|"magazine"|"portrait"|etc.
    language: str                   # Display language code
    date_str: str                   # Formatted date string
    time_str: str                   # Formatted time string

@dataclass
class RenderResult:
    png_path: Path                  # Path to rendered 4K PNG
    html_path: Path                 # Path to rendered HTML (for debugging)
    template_used: str              # Which template was used
    story_count: int                # How many stories were rendered
    painting_hash: str              # Content hash of painting used

def build_template_variables(ctx: RenderContext) -> dict[str, str]:
    """Convert RenderContext into a flat dict of template variables.

    Each key corresponds to a {{PLACEHOLDER}} in the HTML template.
    Builds story HTML blocks, handles orientation-specific variables,
    and resolves the painting image path to a file:// URL.
    """
    ...

async def render_to_png(
    ctx: RenderContext,
    output_dir: Path,
) -> RenderResult:
    """Render a 4K PNG from the given context.

    Steps:
    1. Load template HTML from templates/{template_name}.html
    2. Substitute template variables
    3. Write HTML to output_dir/
    4. Launch Playwright, set viewport to 3840x2160
    5. Screenshot to PNG
    6. Close browser
    7. Return RenderResult

    Raises RenderError if Playwright fails or template is not found.
    """
    ...

def render_to_png_sync(ctx: RenderContext, output_dir: Path) -> RenderResult:
    """Synchronous wrapper around render_to_png."""
    ...
```

---

### 2.8 Module: `tv`

**File**: `src/sfumato/tv.py`

**Responsibility**: Connect to Samsung The Frame TV via WebSocket (samsungtvws library),
check TV status and Art Mode availability, upload PNG images, set the displayed image,
list uploaded images, and clean up old uploads.

**Non-Responsibility**: Does not render images. Does not decide what to display. Does not
manage scheduling.

**Public Interface**:

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TvStatus:
    reachable: bool              # Can we connect at all?
    art_mode_supported: bool     # Does this TV support Art Mode?
    art_mode_active: bool        # Is the TV currently in Art Mode?
    uploaded_count: int          # Number of images currently uploaded
    error: str | None = None     # Error message if reachable is False

@dataclass
class UploadedImage:
    content_id: str              # Samsung's content ID
    file_name: str | None        # Original file name if available

def check_status(tv_config: TvConfig) -> TvStatus:
    """Check TV connectivity and Art Mode status.

    Non-throwing: always returns TvStatus, with reachable=False on failure.
    Timeout: 10 seconds for connection attempt.
    """
    ...

def upload_image(
    tv_config: TvConfig,
    image_path: Path,
) -> str:
    """Upload a PNG to Art Mode. Returns the content_id assigned by the TV.

    Raises TvConnectionError if TV is unreachable.
    Raises TvUploadError if upload fails.
    """
    ...

def set_displayed(tv_config: TvConfig, content_id: str) -> None:
    """Switch the TV to display the specified content_id.

    Raises TvConnectionError if TV is unreachable.
    """
    ...

def list_uploaded(tv_config: TvConfig) -> list[UploadedImage]:
    """List all images currently uploaded to the TV's Art Mode.

    Raises TvConnectionError if TV is unreachable.
    """
    ...

def delete_uploaded(tv_config: TvConfig, content_id: str) -> None:
    """Delete a specific uploaded image from the TV.

    Raises TvConnectionError if TV is unreachable.
    """
    ...

def clean_old_uploads(tv_config: TvConfig, keep: int) -> int:
    """Delete oldest uploads, keeping only the most recent `keep` images.

    Returns the number of images deleted.
    Non-throwing: logs warnings on individual deletion failures.
    """
    ...

def is_available_for_push(tv_config: TvConfig) -> bool:
    """Check if the TV is reachable AND in Art Mode.

    Convenience function combining check_status fields.
    Non-throwing: returns False on any error.
    """
    ...
```

**Error Model**:

```python
class TvError(Exception):
    """Base exception for TV operations."""
    pass

class TvConnectionError(TvError):
    """TV is unreachable or connection was refused."""
    pass

class TvUploadError(TvError):
    """Image upload to TV failed."""
    pass
```

---

### 2.9 Module: `llm`

**File**: `src/sfumato/llm.py`

**Responsibility**: Provide a unified interface for invoking LLM backends (gemini CLI, codex
CLI, claude-code CLI) via subprocess. Handle prompt construction, response parsing, retries,
and timeout.

**Non-Responsibility**: Does not know about paintings, news, or layout semantics. Does not
cache results (callers cache). Purely a transport/invocation layer.

**Public Interface**:

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class LlmResponse:
    text: str                    # Raw text response from the LLM
    model: str                   # Model that generated the response
    cli: str                     # CLI backend used
    usage: dict | None = None    # Token usage if available

async def invoke_text(
    prompt: str,
    ai_config: AiConfig,
    system_prompt: str | None = None,
    max_tokens: int = 4000,
    temperature: float = 0.3,
    timeout_seconds: int = 120,
) -> LlmResponse:
    """Invoke LLM with a text-only prompt.

    Dispatches to the configured CLI backend:
    - gemini: `gemini -m {model} -p "{prompt}"`
    - codex: `codex -m {model} -p "{prompt}"`
    - claude-code: `claude -m {model} -p "{prompt}" --output-format json`

    Retries up to 2 times on transient errors (timeout, connection refused).
    Raises LlmError on persistent failure.
    """
    ...

async def invoke_vision(
    prompt: str,
    image_path: Path,
    ai_config: AiConfig,
    max_tokens: int = 4000,
    timeout_seconds: int = 180,
) -> LlmResponse:
    """Invoke LLM with an image + text prompt.

    Passes the image as a file reference to the CLI backend.
    Used for painting layout analysis.

    Raises LlmError on failure.
    """
    ...

def parse_json_response(text: str) -> dict:
    """Parse LLM response as JSON, stripping markdown code fences if present.

    Handles common LLM formatting issues:
    - ```json ... ``` wrappers
    - Leading/trailing whitespace
    - Trailing commas (lenient parsing)

    Raises LlmParseError if parsing fails.
    """
    ...
```

**Error Model**:

```python
class LlmError(Exception):
    """LLM invocation failed after retries."""
    pass

class LlmParseError(LlmError):
    """LLM response could not be parsed as expected format."""
    pass
```

---

### 2.10 Module: `state`

**File**: `src/sfumato/state.py`

**Responsibility**: Manage all persistent and runtime state for the daemon: news queue,
used paintings set, layout cache, and painting pool metadata. Persist state to disk
(JSON files in `~/.sfumato/state/`) and load on startup.

**Non-Responsibility**: Does not make decisions about what to enqueue or dequeue (that is
the orchestrator and scheduler). Purely a storage layer.

**Public Interface**:

```python
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np

@dataclass
class QueuedBatch:
    stories: list[Story]
    tone_description: str
    enqueued_at: datetime

class NewsQueue:
    """FIFO queue of news story batches, persisted to disk."""

    def __init__(self, state_dir: Path) -> None: ...

    def enqueue(self, result: CurationResult, batch_size: int) -> int:
        """Split CurationResult.stories into batches of batch_size,
        enqueue each batch. Returns number of batches enqueued."""
        ...

    def dequeue(self) -> QueuedBatch | None:
        """Remove and return the next batch. Returns None if empty."""
        ...

    def peek(self) -> QueuedBatch | None:
        """Return the next batch without removing it."""
        ...

    def expire(self, expire_days: int) -> int:
        """Remove batches older than expire_days. Returns count removed."""
        ...

    @property
    def size(self) -> int:
        """Number of batches currently in queue."""
        ...

    def save(self) -> None:
        """Persist queue to disk."""
        ...

    def load(self) -> None:
        """Load queue from disk. No-op if file doesn't exist."""
        ...

class ReplayBatch:
    """A replayable news batch with metadata for cyclic replay."""

    stories: list[Story]
    tone_description: str
    source_enqueued_at: datetime    # When original QueuedBatch was enqueued
    transferred_at: datetime        # When batch entered ReplayQueue
    last_replayed_at: datetime | None  # None until first next() call
    replay_count: int               # Count of completed next() yields

class ReplayQueue:
    """Cyclic replay queue persisted to replay_queue.json.

    When the primary NewsQueue is empty, previously-seen batches cycle
    through on subsequent rotations. Old batches expire based on
    config.news.replay_expire_days.

    Cycling semantics:
    - next() returns the batch at next_index and advances cyclically
    - next() does not remove entries; replay storage is cyclic, not FIFO
    - Empty queue => next() returns None

    Deduplication:
    - _seen_urls tracks all story URLs that have ever been replayed
    - Transfer policy rejects batches with too-high overlap with existing entries
    """

    def __init__(self, state_dir: Path) -> None: ...

    def next(self) -> ReplayBatch | None:
        """Return next replay batch and advance cyclic cursor."""
        ...

    def transfer_from_news_queue(self, batch: QueuedBatch) -> bool:
        """Evaluate overlap and append batch if accepted.

        Uses config.news.replay_expire_days for staleness threshold.
        Returns True if batch was transferred, False if rejected.
        """
        ...

    def expire(self, expire_days: int) -> int:
        """Drop batches older than expire_days. Returns count removed."""
        ...

    @property
    def size(self) -> int:
        """Number of replay batches available."""
        ...

    @property
    def seen_urls(self) -> set[str]:
        """Set of all story URLs that have been replayed (for dedup)."""
        ...

    def persist(self) -> None:
        """Persist queue and seen_urls to disk."""
        ...

    def load(self) -> None:
        """Load queue from disk. No-op if file doesn't exist."""
        ...

class ArtFactRotation:
    """Tracks which art fact to display next for each painting.

    Keys are painting.content_hash values.
    Values are zero-based whisper_fact_index to emit.

    When a painting with art_facts is displayed, the index advances
    modulo len(art_facts) on each successful rotation.
    """

    def __init__(self, state_dir: Path) -> None: ...

    def get_next_index(self, content_hash: str, art_fact_count: int) -> int | None:
        """Return the next fact index for this painting, or None if no facts."""
        ...

    def advance(self, content_hash: str, art_fact_count: int) -> None:
        """Advance the fact index modulo art_fact_count."""
        ...

    def save(self) -> None: ...
    def load(self) -> None: ...

class UsedPaintings:
    """Set of content_hashes that have been displayed. Persisted to disk."""

    def __init__(self, state_dir: Path) -> None: ...

    def mark_used(self, content_hash: str) -> None: ...
    def is_used(self, content_hash: str) -> bool: ...
    def reset(self) -> None:
        """Clear all used marks (when pool is exhausted)."""
        ...
    @property
    def count(self) -> int: ...
    def save(self) -> None: ...
    def load(self) -> None: ...

class LayoutCache:
    """Cache of LayoutParams keyed by painting content_hash. Persisted to disk."""

    def __init__(self, state_dir: Path) -> None: ...

    def get(self, content_hash: str) -> LayoutParams | None: ...
    def put(self, content_hash: str, layout: LayoutParams) -> None: ...
    def has(self, content_hash: str) -> bool: ...
    @property
    def size(self) -> int: ...
    def save(self) -> None: ...
    def load(self) -> None: ...

@dataclass
class AppState:
    """Aggregate root for all daemon state."""
    news_queue: NewsQueue
    replay_queue: ReplayQueue
    used_paintings: UsedPaintings
    layout_cache: LayoutCache
    art_fact_rotation: ArtFactRotation

    @classmethod
    def load(cls, state_dir: Path) -> "AppState":
        """Load all state from state_dir, creating defaults if not present."""
        ...

    def save_all(self) -> None:
        """Persist all state components."""
        ...
```

**State Directory Layout**:

```
~/.sfumato/state/
  news_queue.json        # Serialized QueuedBatch list
  replay_queue.json      # Cyclic replay batches with seen_urls
  used_paintings.json    # Set of content_hash strings
  layout_cache.json      # Dict[content_hash, LayoutParams as dict]
  art_fact_rotation.json # Dict[content_hash, next_fact_index]
```

---

### 2.11 Module: `scheduler`

**File**: `src/sfumato/scheduler.py`

**Responsibility**: Manage the dual-timer architecture for the `watch` daemon. Enforce
`quiet_hours` and `active_hours`. Determine which actions should run on each tick.

**Non-Responsibility**: Does not execute pipeline stages (that is `orchestrator`). Does not
manage state persistence. Purely a timing/scheduling layer.

**Public Interface**:

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, Flag, auto

class Action(Flag):
    NONE = 0
    REFRESH_NEWS = auto()    # Fetch + curate news
    ROTATE = auto()          # Switch painting + render + push
    BACKFILL = auto()        # Background painting pool expansion
    QUIET_ART = auto()       # Display pure art (no news overlay)
    IDLE = auto()            # Outside active hours, do nothing

@dataclass
class SchedulerState:
    last_news_refresh: datetime | None
    last_rotation: datetime | None
    last_backfill: datetime | None

class Scheduler:
    def __init__(self, config: ScheduleConfig) -> None: ...

    def what_to_do(self, now: datetime, state: SchedulerState) -> Action:
        """Determine which actions should run right now.

        Logic:
        1. If now is outside active_hours -> IDLE
        2. If now is within quiet_hours -> QUIET_ART (pure painting, no news)
        3. If news_interval has elapsed since last_news_refresh -> REFRESH_NEWS | ROTATE
        4. If rotate_interval has elapsed since last_rotation -> ROTATE
        5. If pool needs backfill and no other heavy work -> BACKFILL
        6. Else -> NONE

        Multiple actions can be combined (REFRESH_NEWS | ROTATE).
        """
        ...

    def seconds_until_next_action(self, now: datetime, state: SchedulerState) -> float:
        """How many seconds until the next action is due. Used for sleep."""
        ...

    def is_quiet_hour(self, now: datetime) -> bool:
        """Check if current time is within quiet_hours range."""
        ...

    def is_active_hour(self, now: datetime) -> bool:
        """Check if current time is within active_hours range."""
        ...
```

---

### 2.12 Module: `orchestrator`

**File**: `src/sfumato/orchestrator.py`

**Responsibility**: Coordinate the full pipeline for each action. Connect all modules
together: news, paintings, layout_ai, palette, matcher, render, tv. Handle the
"single run" and "watch" execution modes.

**Non-Responsibility**: Does not implement individual stages (those are in their respective
modules). Does not parse CLI arguments (that is `cli`). This is the composition root.

**Public Interface**:

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RunOptions:
    no_upload: bool = False           # Skip TV upload
    no_news: bool = False             # Pure art mode (no news overlay)
    painting_path: Path | None = None # Use specific painting instead of pool
    preview: bool = False             # Open result in system viewer

@dataclass
class RunResult:
    render_result: RenderResult | None    # None if nothing was rendered (e.g., IDLE)
    painting: PaintingInfo | None
    story_count: int
    uploaded: bool
    match_score: float | None             # Semantic similarity score, if applicable
    action: str                           # What action was performed

async def run_once(
    config: AppConfig,
    state: AppState,
    options: RunOptions,
) -> RunResult:
    """Execute a single rotation cycle.

    Pipeline:
    1. If not no_news: dequeue next news batch from state.news_queue
       - If queue is empty, trigger an on-demand refresh_news first
    2. Select painting:
       - If painting_path is specified, use it directly
       - Else if no_news, pick random unused painting
       - Else use matcher.select_painting with news tone
    3. Analyze painting layout (check layout_cache first)
    4. Extract palette (check cache first)
    5. Choose template based on orientation + layout_params.template_hint
    6. Render to 4K PNG
    7. If not no_upload and TV is available: upload and set displayed
    8. Mark painting as used
    9. If preview: open PNG in system viewer
    10. Save state

    Returns RunResult summarizing what happened.
    """
    ...

async def run_news_refresh(
    config: AppConfig,
    state: AppState,
) -> int:
    """Fetch and curate news, enqueue batches.

    Returns number of batches enqueued.
    """
    ...

async def run_backfill(
    config: AppConfig,
    state: AppState,
) -> int:
    """Fetch more paintings to expand the pool toward pool_size.
    Analyze each new painting via LLM (layout + description).

    Returns number of new paintings added.
    """
    ...

async def init_project(config: AppConfig) -> None:
    """Initialize the sfumato project.

    1. Create config file if not present
    2. Create state directory structure
    3. Fetch seed_size paintings from configured sources
    4. Analyze each painting (layout + description)

    This is a potentially long operation (~50 LLM calls for 50 paintings).
    Progress is printed to stdout.
    """
    ...

async def watch(config: AppConfig) -> None:
    """Run the daemon loop.

    1. Load state
    2. Loop:
       a. Ask scheduler what_to_do(now)
       b. Execute the indicated actions
       c. Save state
       d. Sleep until next action
    3. Handle SIGINT/SIGTERM for graceful shutdown
    """
    ...
```

---

### 2.13 Module: `cli`

**File**: `src/sfumato/cli.py`

**Responsibility**: Parse CLI arguments using Typer, load config, and dispatch to
orchestrator functions. Handle top-level error display and exit codes.

**Non-Responsibility**: Does not contain business logic. Does not directly call news, render,
or tv modules (goes through orchestrator).

**Public Interface**:

```python
import typer

app = typer.Typer(help="Turn Samsung The Frame into a living art + news terminal.")

@app.command()
def init(
    config: Path = typer.Option(None, "--config", help="Config file path"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Create config + fetch seed paintings + analyze them."""
    ...

@app.command()
def run(
    config: Path = typer.Option(None, "--config", help="Config file path"),
    no_upload: bool = typer.Option(False, "--no-upload", help="Render only, skip TV upload"),
    no_news: bool = typer.Option(False, "--no-news", help="Pure painting mode"),
    painting: Path = typer.Option(None, "--painting", help="Use specific painting"),
    cli_override: str = typer.Option(None, "--cli", help="Override AI CLI backend"),
    model_override: str = typer.Option(None, "--model", help="Override AI model"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Single execution of the full pipeline."""
    ...

@app.command()
def watch(
    config: Path = typer.Option(None, "--config", help="Config file path"),
    cli_override: str = typer.Option(None, "--cli", help="Override AI CLI backend"),
    model_override: str = typer.Option(None, "--model", help="Override AI model"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Start the daemon (long-running watch mode)."""
    ...

@app.command()
def preview(
    config: Path = typer.Option(None, "--config", help="Config file path"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Render and open in system viewer."""
    ...

# Subcommand group: sfumato tv ...
tv_app = typer.Typer(help="TV management commands.")
app.add_typer(tv_app, name="tv")

@tv_app.command("status")
def tv_status(
    config: Path = typer.Option(None, "--config", help="Config file path"),
) -> None:
    """Check TV connection and Art Mode status."""
    ...

@tv_app.command("list")
def tv_list(
    config: Path = typer.Option(None, "--config", help="Config file path"),
) -> None:
    """List uploaded images on the TV."""
    ...

@tv_app.command("clean")
def tv_clean(
    config: Path = typer.Option(None, "--config", help="Config file path"),
) -> None:
    """Remove old uploads from the TV."""
    ...

def main() -> None:
    """Entry point for `sfumato` command."""
    app()
```

---

## 3. Data Models

All data models are defined as frozen dataclasses (where immutability is appropriate) or
regular dataclasses (where mutation is needed). They are defined in their respective module
files, not in a central `models.py`, because each model belongs to the module responsible
for producing it.

### 3.1 Data Model Summary

| Model            | Defined In     | Produced By        | Consumed By                     |
|------------------|----------------|--------------------|---------------------------------|
| `AppConfig`      | `config.py`    | `load_config()`    | All modules                     |
| `FeedConfig`     | `config.py`    | `load_config()`    | `news`                          |
| `Story`          | `news.py`      | `curate()`         | `state`, `render`, `matcher`    |
| `CurationResult` | `news.py`      | `curate()`         | `state.NewsQueue`               |
| `PaintingInfo`   | `paintings.py` | `fetch_paintings()`| `layout_ai`, `render`, `matcher`|
| `PaletteColors`  | `palette.py`   | `extract_palette()`| `render`                        |
| `LayoutParams`   | `layout_ai.py` | `analyze_painting()`| `render`, `orchestrator`       |
| `RenderContext`  | `render.py`    | `orchestrator`     | `render.render_to_png()`        |
| `RenderResult`   | `render.py`    | `render_to_png()`  | `orchestrator`, `tv`            |
| `TvStatus`       | `tv.py`        | `check_status()`   | `cli`, `orchestrator`           |
| `QueuedBatch`    | `state.py`     | `NewsQueue`        | `orchestrator`                  |
| `AppState`       | `state.py`     | `AppState.load()`  | `orchestrator`                  |
| `RunResult`      | `orchestrator` | `run_once()`       | `cli`                           |
| `LlmResponse`    | `llm.py`       | `invoke_text()`    | `news`, `layout_ai`            |

### 3.2 Data Flow Through Pipeline

```
          CurationResult                PaintingInfo
               │                             │
               ▼                             ▼
     ┌───── stories ──────┐       ┌──── image_path ────┐
     │                    │       │                     │
     │  NewsQueue         │       │  layout_ai          │
     │  .enqueue()        │       │  .analyze_painting()│
     │  .dequeue() ───┐   │       │       │             │
     │                │   │       │       ▼             │
     │                │   │       │  LayoutParams       │
     │                │   │       │       │             │
     │                │   │       │  palette            │
     │                │   │       │  .extract_palette() │
     │                │   │       │       │             │
     │                │   │       │       ▼             │
     │                │   │       │  PaletteColors      │
     │                ▼   ▼       ▼       │             │
     │           ┌─────────────────┐      │             │
     │           │  RenderContext  │◀─────┘             │
     │           └────────┬────────┘                    │
     │                    │                             │
     │                    ▼                             │
     │           ┌─────────────────┐                    │
     │           │  render_to_png  │                    │
     │           └────────┬────────┘                    │
     │                    │                             │
     │                    ▼                             │
     │           ┌─────────────────┐                    │
     │           │  RenderResult   │                    │
     │           │  .png_path ─────┼──── tv.upload()    │
     │           └─────────────────┘                    │
     └──────────────────────────────────────────────────┘
```

### 3.3 Serialization

All persisted data models must be serializable to JSON.
The serialization format for each:

| Model          | Format                  | Notes                                    |
|----------------|-------------------------|------------------------------------------|
| `Story`        | JSON dict               | `published_at` as ISO 8601 string        |
| `QueuedBatch`  | JSON dict               | `enqueued_at` as ISO 8601 string         |
| `LayoutParams` | JSON dict               | All fields are JSON-native except None    |
| `PaintingInfo` | JSON sidecar file       | `image_path` stored as relative to cache |
| `UsedPaintings`| JSON list of strings    | Set of content_hash values               |

---

## 4. External Integrations

### 4.1 RSS Feeds

| Aspect              | Detail                                             |
|---------------------|----------------------------------------------------|
| **Library**         | `feedparser` for parsing, `httpx` for fetching     |
| **Concurrency**     | `httpx.AsyncClient` with concurrent requests       |
| **Timeout**         | 15 seconds per feed                                |
| **Error handling**  | Individual feed failures are logged and skipped     |
| **Rate limiting**   | None needed; feeds are fetched at most every 6h    |
| **Deduplication**   | By article URL across all feeds in a single fetch  |
| **Age filtering**   | Articles older than `max_age_days` are dropped     |


### 4.2 Met Museum API

| Aspect              | Detail                                             |
|---------------------|----------------------------------------------------|
| **Base URL**        | `https://collectionapi.metmuseum.org/public/collection/v1` |
| **Auth**            | None required                                      |
| **Rate limit**      | 80 requests/second (generous)                      |
| **Filtering**       | `isPublicDomain=true`, `medium=Paintings`          |
| **Image download**  | Use `primaryImage` field URL                       |
| **Error handling**  | Retry 2x on 5xx, skip on 404                      |
| **Two-step fetch**  | First `/search` for IDs, then `/objects/{id}` for each |

### 4.4 Wikimedia Commons

| Aspect              | Detail                                             |
|---------------------|----------------------------------------------------|
| **API**             | MediaWiki API or direct category scraping          |
| **Auth**            | None                                               |
| **Rate limit**      | Respect `Retry-After` headers                      |
| **Filtering**       | `Category:Featured_pictures_of_paintings`          |
| **Image download**  | Use `imageinfo` API for original file URL          |
| **User-Agent**      | Must set a descriptive User-Agent per policy       |

### 4.5 Samsung TV (samsungtvws)

| Aspect              | Detail                                             |
|---------------------|----------------------------------------------------|
| **Library**         | `samsungtvws[async,encrypted]`                     |
| **Connection**      | WebSocket to `ws://{ip}:{port}/api/v2/channels/samsung.remote.control` |
| **Auth**            | First connection requires TV confirmation (one-time)|
| **Token storage**   | samsungtvws stores token in `~/.samsungtvws/token` |
| **Art Mode API**    | `tv.art()` -> `.upload()`, `.available()`, `.set_active()` |
| **Timeout**         | 10 second connection timeout                       |
| **Error handling**  | TV unreachable -> skip push (non-fatal)            |
| **Concurrency**     | Single connection; no concurrent operations        |

### 4.6 LLM CLI Backends

| Backend        | Command Pattern                                              | Vision Support |
|----------------|--------------------------------------------------------------|----------------|
| `gemini`       | `gemini -m {model} -p "{prompt}"` / `--image {path}`        | Yes            |
| `codex`        | `codex -m {model} -p "{prompt}"`                             | TBD            |
| `claude-code`  | `claude -m {model} -p "{prompt}" --output-format json`       | Yes            |

| Aspect              | Detail                                             |
|---------------------|----------------------------------------------------|
| **Invocation**      | `subprocess.create_subprocess_exec` (async)        |
| **Timeout**         | 120s for text, 180s for vision                     |
| **Retry**           | 2 retries on timeout or non-zero exit code         |
| **Output parsing**  | Capture stdout, strip markdown fences, parse JSON  |
| **Error handling**  | Raise `LlmError` after retries exhausted           |
| **Concurrency**     | One LLM call at a time (no parallelism)            |

---

## 5. State Management

### 5.1 State Overview

The daemon maintains five categories of state:

| State               | Lifetime        | Storage              | Size Estimate        |
|---------------------|-----------------|----------------------|----------------------|
| **News queue**      | Days            | `news_queue.json`    | All non-spam stories/refresh |
| **Used paintings**  | Until pool exhausted | `used_paintings.json` | ~200 hashes     |
| **Layout cache**    | Forever         | `layout_cache.json`  | ~2KB per painting    |
| **Painting pool**   | Forever         | `paintings/` dir     | ~2MB per image       |

### 5.2 News Queue Lifecycle

```
refresh_news()                          rotate()
     │                                      │
     ▼                                      ▼
┌──────────────────────────────────────────────────┐
│ Queue:  [batch_0] [batch_1] [batch_2] [batch_3]  │
│          ▲                                   │   │
│          │               dequeue ◀───────────┘   │
│     enqueue (split CurationResult                │
│      into batches of recommended_stories)        │
└──────────────────────────────────────────────────┘
     │
     │  expire_days exceeded?
     ▼
  [removed]
```

- All non-spam stories are kept (LLM only filters obvious spam), processed in batches of 15
- They are queued in batches of 8 stories; each rotation renders the count determined by `recommended_stories` (2-7)
- The batch size is dynamic: it depends on the selected painting's layout analysis
- If the queue is empty when a rotation triggers, an on-demand news refresh is performed

### 5.3 Painting Pool Lifecycle

```
init()                          watch() background
  │                                 │
  ▼                                 ▼
Fetch seed_size (50)          Backfill to pool_size (200)
  │                                 │
  ▼                                 ▼
Analyze each (LLM)            Analyze each (LLM)
  │                                 │
  ▼                                 ▼
Store in cache_dir            Store in cache_dir
```

- Paintings are fetched from configured sources in round-robin
- Each painting gets: sidecar JSON metadata, LLM layout analysis (cached)
- `UsedPaintings` tracks which have been displayed; when all are used, the set resets

### 5.4 State Persistence Strategy

- State is saved to disk after every action (rotation, news refresh, backfill)
- State is loaded once on daemon startup
- No database; plain files in `~/.sfumato/state/`
- File writes are atomic: write to temp file, then `os.replace()` to target
- Concurrent access is not supported (single daemon instance assumed)

### 5.5 Cache Invalidation

| Cache             | Invalidation Policy                              |
|-------------------|--------------------------------------------------|
| Layout cache      | Never invalidated (keyed by content hash)        |
| News queue        | Entries older than `expire_days` are purged      |
| Used paintings    | Reset when all paintings in pool have been used  |
| Painting sidecar  | Never invalidated (immutable once written)       |

---

## 6. Scheduling Mechanism

### 6.1 Timer Design

The `watch` command uses a single-threaded async event loop with calculated sleeps:

```python
# Pseudocode for watch loop (NOT implementation -- for illustration only)
while running:
    now = datetime.now()
    action = scheduler.what_to_do(now, scheduler_state)

    if Action.REFRESH_NEWS in action:
        await run_news_refresh(config, state)
        scheduler_state.last_news_refresh = now

    if Action.ROTATE in action:
        await run_once(config, state, RunOptions())
        scheduler_state.last_rotation = now

    if Action.BACKFILL in action:
        await run_backfill(config, state)
        scheduler_state.last_backfill = now

    if Action.QUIET_ART in action:
        await run_once(config, state, RunOptions(no_news=True))
        scheduler_state.last_rotation = now

    state.save_all()
    sleep_seconds = scheduler.seconds_until_next_action(now, scheduler_state)
    await asyncio.sleep(sleep_seconds)
```

### 6.2 Time Window Logic

```
Hour:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
       ├──quiet_hours──┤  │  ├──────────────active_hours───────────────────┤
       │               │  │  │                                             │
       │  IDLE         │  │  │  Normal operation:                          │
       │  (no push,    │  │  │  ROTATE every 15min                         │
       │   no render)  │  │  │  REFRESH_NEWS every 6h                      │
       │               │  6  │  BACKFILL when pool < pool_size             │
       │               │  │  │                                             │
       └───────────────┘  │  └─────────────────────────────────────────────┘
                          │
                    transition: first action of the day
                    may trigger immediate REFRESH_NEWS + ROTATE
```

- **quiet_hours `[0, 6)`**: TV may still be on in Art Mode. If configured, push a pure
  painting (no news). Otherwise, do nothing.
- **active_hours `[7, 23]`**: Full operation. Rotations and news refreshes happen here.
- **Outside both**: IDLE. No rendering, no pushing, no network calls.
- **Edge case**: If the daemon starts during quiet_hours or outside active_hours, it
  initializes state but does not act until active_hours begin.

### 6.3 Startup Behavior

On daemon startup:

1. Load all state from disk
2. Expire old news queue entries
3. Check if news refresh is overdue -> if yes, refresh immediately
4. Check scheduler -> execute first action
5. Enter main loop

---

## 7. LLM Invocation Strategy

### 7.1 When LLM Is Called

| Operation               | LLM Type | Frequency         | Cacheable? | Cost     |
|--------------------------|----------|--------------------|------------|----------|
| News curation            | Text     | Every 6h           | No         | ~12 stories |
| Painting layout analysis | Vision   | Once per painting  | Yes (forever) | ~1 image |
| Painting description     | Vision   | Once per painting  | Yes (forever) | Part of layout call |
| Painting-news matching   | Text     | Every rotation     | No (unique) | ~1 prompt |

### 7.2 Prompt Structure

#### News Curation Prompt

```
System: You are a multilingual news editor curating a visual briefing.

Input:
- Raw RSS entries (title, summary, source, category, published date), processed in batches of 15
- Target language: {language}
- Keep all stories, only filter obvious spam

Output (JSON):
- stories: [{headline, summary, source, category, url, featured}]
- tone_description: "free-form description of the batch's emotional atmosphere"

Constraints:
- Headlines: concise, max 15 chars (CJK) or 12 words (Latin)
- Summaries: 60-100 chars (CJK) or 60-100 words (Latin), complete and informative
- Mark exactly ONE story as featured
- Output entirely in {language}
- tone_description: 2-3 sentences capturing the mood, themes, tension of this batch
```

#### Painting Analysis Prompt

```
System: You are a visual composition analyst for a 3840x2160 display.

Input:
- Painting image (attached)

Output (JSON):
- orientation: "landscape" | "portrait"
- painting_title: identified title or "Unknown"
- painting_artist: identified artist or "Unknown"
- painting_description: rich free-form description (mood, themes, atmosphere, colors)
- text_zone: {position, reason}
- colors: {text_primary, text_secondary, text_dim, text_shadow, scrim_color, panel_bg, border, accent}
- scrim: {position_css, size_css, gradient_css}
- recommended_stories: 2-5
- template_hint: "painting_text" | "magazine" | "portrait"
- portrait_layout: null | {painting_width_percent, left_panel_color, right_panel_color, info_side}

Constraints:
- Text position: at least 120px from edges
- Text max width: <=45% of screen width (landscape)
- Scrim: radial-gradient, subtle (not a visible box)
- Colors: harmonize with the painting
- recommended_stories: based on available quiet zone size
```

### 7.3 Unified CLI Backend Interface

All LLM calls go through `llm.py`, which dispatches to the configured CLI:

```
                    ┌──────────┐
                    │  llm.py  │
                    │ invoke() │
                    └────┬─────┘
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
       ┌─────────┐ ┌─────────┐ ┌──────────┐
       │ gemini  │ │ codex   │ │ claude   │
       │ CLI     │ │ CLI     │ │ CLI      │
       └─────────┘ └─────────┘ └──────────┘
            │            │            │
            └────────────┼────────────┘
                         ▼
                    subprocess
```

### 7.4 Caching Strategy

- **Layout analysis**: Cached by `content_hash` in `LayoutCache`. One LLM vision call per
  unique painting, forever. This is the most expensive call (vision model) and the most
  cacheable (painting content never changes).
- **Painting description**: Part of the layout analysis response. Cached together.
- **News curation**: NOT cached. Each refresh produces fresh content from fresh RSS entries.
- **Painting-news matching**: NOT cached. Each rotation compares news tone with painting descriptions via LLM.

### 7.5 Cost Optimization

1. **Batch all curation into one LLM call**: All raw entries go in, all curated stories + tone
   come out. One call per 6h refresh.
2. **Combine layout + description**: The painting analysis prompt asks for both layout params
   and the free-form description. One vision call per painting.
3. **Cache aggressively**: Layout and description are immutable per painting hash. Even if the
   config changes, cached values remain valid.
4. **Backfill during idle time**: Painting analysis is done during BACKFILL actions, which
   are lower priority than ROTATE and REFRESH.

---

## 8. Template System

### 8.1 Template Selection Logic

```
                         ┌──────────────────┐
                         │ Is no_news mode? │
                         └────────┬─────────┘
                                  │
                    ┌─────────yes─┤──no─────────────────┐
                    ▼                                     ▼
           ┌────────────────┐                    ┌────────────────┐
           │ Pure art mode  │                    │ orientation?   │
           │ (no template)  │                    └───────┬────────┘
           │ Push raw image │              ┌─landscape──┤──portrait──┐
           └────────────────┘              ▼                         ▼
                               ┌───────────────────┐     ┌──────────────────┐
                               │ Has quiet zones?  │     │ portrait.html    │
                               │ (from layout_ai)  │     │ Three-panel      │
                               └──────┬────────────┘     │ layout           │
                          ┌───yes─────┤─────no────┐      └──────────────────┘
                          ▼                       ▼
                ┌──────────────────┐    ┌──────────────────┐
                │ painting_text    │    │ magazine.html    │
                │ Text blended     │    │ 72/28 split      │
                │ into painting    │    │ Painting | Text  │
                └──────────────────┘    └──────────────────┘
```

Decision factors:
- **`painting_text.html`**: Landscape painting with identified quiet zones (sky, water, fog).
  Text overlays directly on the painting with localized scrim.
- **`magazine.html`**: Landscape painting with dense composition (no good quiet zones). Fixed
  72/28 horizontal split: painting left, text panel right.
- **`portrait.html`**: Portrait-oriented painting. Three-panel layout: left info panel,
  center painting, right news panel.
- **`art_overlay.html`**: Alternative landscape overlay with glass-morphism effect. Used when
  painting_text would obscure too much of the subject.
- **`art_minimal.html`**: Minimal bottom band overlay. Used for paintings with important
  bottom areas (when painting_text targets top).
- **`gallery_wall.html`**: Painting displayed as if on a gallery wall with frame, news on
  the side. Used for paintings that look best with a visible "frame" treatment.
- **`news_poster.html`**: Legacy/fallback pure-news template (no painting). Three-column
  dark theme. Used only if no painting is available.

### 8.2 Template Variable Contracts

Each template uses `{{PLACEHOLDER}}` syntax. The `render` module's `build_template_variables()`
function produces the appropriate variables for each template.

#### `painting_text.html`

| Variable          | Type    | Source                | Example                              |
|-------------------|---------|-----------------------|--------------------------------------|
| `{{BG_IMAGE}}`    | URL     | `file://{image_path}` | `file:///home/user/.sfumato/...jpg`  |
| `{{SCRIM_GRADIENT}}`| CSS  | `layout.scrim.gradient_css` | `radial-gradient(...)` — applied as text-zone background |
| `{{TEXT_POSITION}}`| CSS    | `layout.text_zone` -> CSS | `top: 100px; right: 160px;`      |
| `{{TEXT_WIDTH}}`   | CSS    | `layout` derived      | `1500px`                             |
| `{{TEXT_COLOR}}`   | Color  | `layout.colors.text_primary`| `#F5F0EB`                       |
| `{{TEXT_COLOR_SEC}}`| Color | `layout.colors.text_secondary`| `rgba(245,240,235,0.7)`       |
| `{{TEXT_COLOR_DIM}}`| Color | `layout.colors.text_dim`| `rgba(245,240,235,0.5)`             |
| `{{TEXT_SHADOW}}`  | CSS    | `layout.colors.text_shadow`| `0 1px 8px rgba(0,0,0,0.6)`     |
| `{{DATELINE}}`     | String | Formatted date+time   | `2026-03-14 SAT 08:00`               |
| `{{NEWS_BLOCKS}}`  | HTML   | Built from stories    | `<div class="news-block">...</div>`  |

#### `magazine.html`

| Variable          | Type    | Source                | Example                              |
|-------------------|---------|-----------------------|--------------------------------------|
| `{{BG_IMAGE}}`    | URL     | `file://{image_path}` |                                      |
| `{{PANEL_BG}}`    | Color   | `layout.colors.panel_bg` | `#1a1814`                         |
| `{{TEXT}}`         | Color   | `layout.colors.text_primary` | `#F5F0EB`                    |
| `{{TEXT_SEC}}`     | Color   | `layout.colors.text_secondary` |                              |
| `{{DIM}}`          | Color   | `layout.colors.text_dim` |                                    |
| `{{BORDER}}`       | Color   | `layout.colors.border` | `rgba(255,255,255,0.1)`             |
| `{{DATE}}`         | String  | Formatted date        |                                      |
| `{{SOURCE}}`       | String  | "sfumato" or similar  |                                      |
| `{{STORIES}}`      | HTML    | Story blocks          |                                      |
| `{{ARTIST}}`       | String  | `painting.artist`     |                                      |
| `{{PAINTING_TITLE}}`| String | `painting.title`     |                                      |

#### `portrait.html`

| Variable              | Type    | Source                     |
|------------------------|---------|----------------------------|
| `{{BG_COLOR}}`         | Color   | `layout.colors.panel_bg`   |
| `{{LEFT_WIDTH}}`       | CSS     | Derived from `portrait_layout.painting_width_percent` |
| `{{PAINTING_WIDTH}}`   | CSS     | `{painting_width_percent}%` |
| `{{PAINTING_SRC}}`     | URL     | `file://{image_path}`      |
| `{{TEXT}}`             | Color   | `layout.colors.text_primary` |
| `{{TEXT_SEC}}`         | Color   | `layout.colors.text_secondary` |
| `{{DIM}}`              | Color   | `layout.colors.text_dim`   |
| `{{ACCENT}}`           | Color   | `layout.colors.accent`     |
| `{{BORDER}}`           | Color   | `layout.colors.border`     |
| `{{DATE}}`             | String  |                            |
| `{{SOURCE}}`           | String  |                            |
| `{{ARTIST}}`           | String  |                            |
| `{{PAINTING_TITLE}}`   | String  |                            |
| `{{PAINTING_YEAR}}`    | String  |                            |
| `{{SECTION_LABEL}}`    | String  | e.g. "TODAY'S BRIEFING"    |
| `{{STORIES}}`          | HTML    | Story blocks               |

#### `art_overlay.html`

| Variable          | Type    | Source                     |
|-------------------|---------|-----------------------------|
| `{{BG_IMAGE}}`    | URL     | `file://{image_path}`       |
| `{{TITLE}}`       | String  | e.g. "DAILY BRIEF"         |
| `{{DATE}}`        | String  | Formatted date              |
| `{{UPDATE_TIME}}` | String  | Formatted time              |
| `{{STORIES}}`     | HTML    | Story cards with glass effect |
| `{{ART_CREDIT}}`  | String  | "Artist -- Title (Year)"    |

#### `art_minimal.html`

| Variable          | Type    | Source                     |
|-------------------|---------|-----------------------------|
| `{{BG_IMAGE}}`    | URL     | `file://{image_path}`       |
| `{{TITLE}}`       | String  |                             |
| `{{DATE}}`        | String  |                             |
| `{{UPDATE_TIME}}` | String  |                             |
| `{{STORIES}}`     | HTML    | Grid of 4 story blocks      |
| `{{ART_CREDIT}}`  | String  |                             |

#### `gallery_wall.html`

| Variable              | Type    | Source                     |
|------------------------|---------|----------------------------|
| `{{WALL_COLOR}}`       | Color   | From palette.background    |
| `{{FRAME_COLOR}}`      | Color   | Derived from palette       |
| `{{FRAME_INNER}}`      | Color   |                            |
| `{{FRAME_OUTER}}`      | Color   |                            |
| `{{TEXT_PRI}}`         | Color   |                            |
| `{{TEXT_SEC}}`         | Color   |                            |
| `{{TEXT_DIM}}`         | Color   |                            |
| `{{ACCENT}}`           | Color   |                            |
| `{{BORDER_COLOR}}`     | Color   |                            |
| `{{PAINTING_SRC}}`     | URL     |                            |
| `{{PAINTING_W}}`       | String  | Image width in pixels      |
| `{{PAINTING_H}}`       | String  | Image height in pixels     |
| `{{ARTIST}}`           | String  |                            |
| `{{PAINTING_TITLE}}`   | String  |                            |
| `{{DATE}}`             | String  |                            |
| `{{WEATHER}}`          | String  |                            |
| `{{UPDATE_TIME}}`      | String  |                            |
| `{{NEWS_ITEMS}}`       | HTML    |                            |

#### `news_poster.html` (legacy fallback)

| Variable          | Type    | Source                     |
|-------------------|---------|-----------------------------|
| `{{TITLE}}`       | String  | "MORNING BRIEF"             |
| `{{DATE}}`        | String  |                             |
| `{{UPDATE_TIME}}` | String  |                             |
| `{{WEATHER}}`     | String  |                             |
| `{{TEMP}}`        | String  |                             |
| `{{COLUMNS}}`     | HTML    | Three column divs           |

### 8.3 Story HTML Block Format

For templates using individual story blocks (`{{STORIES}}`, `{{NEWS_BLOCKS}}`, `{{NEWS_ITEMS}}`),
each story is rendered as:

```html
<!-- For magazine.html, portrait.html -->
<div class="story">
  <div class="headline">{story.headline}</div>
  <div class="body">{story.summary}</div>
</div>

<!-- For painting_text.html -->
<div class="news-block">
  <div class="title">{story.headline}</div>
  <div class="body">{story.summary}</div>
</div>

<!-- For art_overlay.html -->
<div class="story">
  <div class="category">{story.category}</div>
  <div class="headline">{story.headline}</div>
  <div class="summary">{story.summary}</div>
  <div class="source">{story.source}</div>
</div>

<!-- For gallery_wall.html -->
<div class="news-item">
  <div class="category">{story.category}</div>
  <div class="headline">{story.headline}</div>
  <div class="summary">{story.summary}</div>
  <div class="source">{story.source}</div>
</div>
<div class="news-divider"></div>
```

---

## 9. Deployment Architecture

### 9.1 Dockerfile Contract

The repository-level `Dockerfile` is a contract stub in this phase. The production image must
conform to these stage boundaries:

- `base-runtime`: Python runtime plus Playwright-compatible shared libraries and multilingual fonts
- `builder`: application artifact assembly from `pyproject.toml`, `src/`, and `templates/`
- `runtime`: final daemon image exposing `sfumato watch` with `/data`-based config and state

The contract artifact lives in `Dockerfile`; rationale and failure conditions are captured in
`DEPLOYMENT_CONTRACT.md`.

### 9.2 Environment Variables

| Variable             | Purpose                              | Required |
|----------------------|--------------------------------------|----------|
| `SFUMATO_CONFIG`     | Path to config.toml                  | No (has defaults) |
| `SFUMATO_DATA_DIR`   | Override `~/.sfumato` data directory | No       |

### 9.3 Data Volumes

```
/data/                          # Mounted volume
  config.toml                   # User configuration
  paintings/                    # Painting cache (persistent)
    met/
    wikimedia/
  state/                        # Daemon state (persistent)
    news_queue.json
    used_paintings.json
    layout_cache.json
  output/                       # Rendered PNGs (transient, can be cleaned)
```

### 9.4 Docker Compose Contract

The repository-level `docker-compose.yml` is a contract stub in this phase. The production
compose definition must provide:

- a single long-running `sfumato` service targeting the Docker `runtime` stage
- `/data/config.toml`, `/data/paintings`, `/data/state`, and optional `/data/output` mounts
- `SIGTERM`-based shutdown with a non-zero grace period
- a healthcheck tied to daemon freshness rather than raw PID existence
- host-network deployment by default on Linux hosts

Note: `network_mode: host` is required because Samsung TV communication uses WebSocket
on the local network. Bridge networking would require the TV IP to be routable from the
container's network namespace.

### 9.5 Container Process Management

- Single process: `sfumato watch` runs as PID 1
- Graceful shutdown: Handles `SIGTERM` (from `docker stop`) by finishing current action,
  saving state, then exiting
- No process supervisor needed (no multiple processes)
- Health check contract: Expose a file-based freshness indicator at
  `/data/state/last_action.json` (or an explicitly superseding path) and treat stale timestamps as
  unhealthy

---

## 10. Error Handling

### 10.1 Error Categories and Strategies

| Error                    | Severity  | Strategy                                    |
|--------------------------|-----------|---------------------------------------------|
| RSS feed unreachable     | Warning   | Skip feed, continue with others             |
| All RSS feeds fail       | Error     | Skip news refresh, retry next interval      |
| LLM call fails           | Error     | Retry 2x, then skip current action          |
| LLM response unparseable | Error     | Retry once, then skip                       |
| Painting download fails  | Warning   | Skip painting, continue with others         |
| All painting sources fail| Error     | Use cached paintings only                   |
| Palette extraction fails | Warning   | Use default palette (neutral grays)         |
| Template not found       | Fatal     | Raise, should never happen in production    |
| Playwright fails         | Error     | Retry once, log error, skip rotation        |
| TV unreachable           | Warning   | Skip upload, save PNG locally               |
| TV upload fails          | Warning   | Skip upload, save PNG locally               |
| Config file invalid      | Fatal     | Print error, exit with code 1               |
| State file corrupted     | Error     | Log warning, reinitialize that state component |
| Disk full                | Fatal     | Log error, exit                             |
| LLM matching fails       | Warning   | Fall back to random painting selection      |

### 10.2 Degradation Ladder

The system degrades gracefully through these levels:

```
Level 0: Full operation
  ├── News curated, painting matched semantically, uploaded to TV
  │
Level 1: No TV (TV unreachable or not in Art Mode)
  ├── Everything works, PNG saved locally, not uploaded
  │
Level 2: No semantic matching (LLM matching fails or too few candidates)
  ├── Random painting selection, everything else works
  │
Level 3: No fresh news (all RSS feeds fail)
  ├── Re-display last successful batch, or pure art mode
  │
Level 4: No LLM (all LLM calls fail)
  ├── Use cached layouts for known paintings
  ├── Skip news curation, display pure art
  │
Level 5: No network at all
  ├── Use cached paintings + cached layouts
  ├── Pure art mode with cached content
  │
Level 6: Fatal (no config, no state, disk full)
  └── Exit with error message
```

### 10.3 Exception Hierarchy

```
SfumatoError (base)
├── ConfigError              # Config parsing/validation failures
├── LlmError                 # LLM invocation failures
│   └── LlmParseError       # LLM response parsing failures
├── TvError                  # TV communication failures
│   ├── TvConnectionError    # TV unreachable
│   └── TvUploadError        # Upload specifically failed
├── RenderError              # Playwright/template failures
├── MatcherError             # No paintings available for matching
├── PaintingFetchError       # Art source API failures
└── StateError               # State file corruption
```

### 10.4 Logging Strategy

- Use Python's `logging` module
- Default level: `INFO` (shows actions taken, skipped, errors)
- `-v/--verbose`: `DEBUG` (shows LLM prompts, HTTP responses, timing)
- Log format: `%(asctime)s %(levelname)s %(name)s: %(message)s`
- Each module uses `logging.getLogger(__name__)`
- Daemon mode: logs to stdout (Docker captures it)

---

## 11. Directory Structure

```
sfumato/
├── ARCHITECTURE.md              # This document
├── README.md                    # User-facing documentation
├── pyproject.toml               # Package metadata and dependencies
├── Dockerfile                   # Container build
├── docker-compose.yml           # Example deployment
├── templates/                   # HTML templates (static, shipped with package)
│   ├── painting_text.html       # Landscape with quiet zones -- text on painting
│   ├── magazine.html            # Landscape dense -- 72/28 split
│   ├── portrait.html            # Portrait -- three-panel layout
│   ├── art_overlay.html         # Landscape -- glass-morphism overlay
│   ├── art_minimal.html         # Landscape -- bottom band overlay
│   ├── gallery_wall.html        # Gallery frame treatment
│   └── news_poster.html         # Legacy pure-news fallback
├── src/
│   └── sfumato/
│       ├── __init__.py          # Package version, SfumatoError base class
│       ├── __main__.py          # `python -m sfumato` entry
│       ├── cli.py               # Typer CLI commands
│       ├── config.py            # TOML config loading, AppConfig dataclass
│       ├── news.py              # RSS fetch + LLM curation
│       ├── paintings.py         # Cloud art sources + local cache management
│       ├── palette.py           # Color extraction from paintings
│       ├── layout_ai.py         # LLM painting analysis
│       ├── matcher.py           # Semantic painting-news matching
│       ├── render.py            # HTML template rendering to 4K PNG
│       ├── tv.py                # Samsung TV control
│       ├── llm.py               # Unified LLM CLI backend interface
│       ├── state.py             # State management (queues, caches)
│       ├── scheduler.py         # Timer logic, time windows
│       └── orchestrator.py      # Pipeline coordination
└── tests/
    ├── test_config.py
    ├── test_news.py
    ├── test_paintings.py
    ├── test_palette.py
    ├── test_layout_ai.py
    ├── test_matcher.py
    ├── test_render.py
    ├── test_tv.py
    ├── test_llm.py
    ├── test_state.py
    ├── test_scheduler.py
    ├── test_orchestrator.py
    └── conftest.py              # Shared fixtures
```

### Runtime Data Layout

```
~/.sfumato/                      # Default data directory (configurable)
├── paintings/                   # Painting cache
│   ├── met/
│   │   ├── 436535.jpg
│   │   └── 436535.json
│   └── wikimedia/
│       ├── Mona_Lisa.jpg
│       └── Mona_Lisa.json
├── state/
│   ├── news_queue.json
│   ├── used_paintings.json
│   └── layout_cache.json
└── output/                      # Rendered images (transient)
    ├── 20260314_080000.png
    └── 20260314_080000.html     # Debug HTML
```

---

## 12. Dependency Graph

### 12.1 Module Dependency Diagram

```
                    ┌─────────┐
                    │   cli   │
                    └────┬────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ orchestrator │
                  └──────┬───────┘
                         │
         ┌───────┬───────┼───────┬───────┬────────┐
         ▼       ▼       ▼       ▼       ▼        ▼
     ┌──────┐┌──────┐┌───────┐┌──────┐┌───────┐┌──────────┐
     │ news ││paint-││matcher││render││  tv   ││scheduler │
     │      ││ings  ││       ││      ││       ││          │
     └──┬───┘└──┬───┘└───┬───┘└──┬───┘└───┬───┘└──────────┘
        │       │        │       │        │
        ▼       ▼        ▼       ▼        ▼
     ┌──────────────────────────────────────┐
     │              llm.py                  │
     │  (subprocess: gemini/codex/claude)   │
     └──────────────────────────────────────┘
        │       │                │        │
        ▼       ▼                ▼        ▼
     ┌──────────────────────────────────────┐
     │            config.py                 │
     │         (AppConfig, all *Config)     │
     └──────────────────────────────────────┘
                         │
                         ▼
     ┌──────────────────────────────────────┐
     │            state.py                  │
     │  (NewsQueue, LayoutCache, etc.)      │
     └──────────────────────────────────────┘

     Additional dependencies:
       render   ──▶ palette.py (PaletteColors)
       render   ──▶ layout_ai.py (LayoutParams -- data only, not calling)
       news     ──▶ llm.py (curate via LLM)
       layout_ai──▶ llm.py (analyze via vision LLM)
       matcher  ──▶ llm.py (LLM-based painting selection)
       paintings──▶ httpx (cloud API calls)
```

### 12.2 Dependency Table

| Module        | Depends On                          | Depended By                | Coupling |
|---------------|-------------------------------------|----------------------------|----------|
| `cli`         | `config`, `orchestrator`            | (entry point)              | Low      |
| `orchestrator`| `config`, `state`, `news`, `paintings`, `layout_ai`, `palette`, `matcher`, `render`, `tv`, `scheduler` | `cli` | High (by design) |
| `news`        | `config`, `llm`                     | `orchestrator`             | Low      |
| `paintings`   | `config`, `httpx`, `PIL`            | `orchestrator`, `matcher`  | Low      |
| `layout_ai`   | `config`, `llm`                     | `orchestrator`             | Low      |
| `palette`     | `PIL`, `numpy`                      | `render`, `orchestrator`   | Low      |
| `matcher`     | `config`, `llm`                     | `orchestrator`             | Low      |
| `render`      | `config`, `playwright`              | `orchestrator`             | Low      |
| `tv`          | `config`, `samsungtvws`             | `orchestrator`             | Low      |
| `llm`         | `config`, `asyncio.subprocess`      | `news`, `layout_ai`, `matcher` | Low  |
| `state`       | (none)                              | `orchestrator`             | Low      |
| `scheduler`   | `config`                            | `orchestrator`             | Low      |
| `config`      | `tomllib` (stdlib)                  | All modules                | Low      |

### 12.3 Dependency Direction

All dependencies flow inward toward shared infrastructure:

```
CLI layer:        cli
                   │
Coordination:     orchestrator, scheduler
                   │
Domain modules:   news, paintings, layout_ai, palette, matcher, render, tv
                   │
Infrastructure:   llm, state, config
```

There are NO circular dependencies. Domain modules do not depend on each other
(they communicate only through the orchestrator via data models). Infrastructure
modules (`llm`, `state`, `config`) are pure utilities with no upward dependencies.

### 12.4 External Dependency Summary

| Package                        | Used By          | Purpose                     |
|--------------------------------|------------------|-----------------------------|
| `typer>=0.9`                   | `cli`            | CLI framework               |
| `feedparser`                   | `news`           | RSS parsing                 |
| `httpx`                        | `news`, `paintings` | HTTP client              |
| `samsungtvws[async,encrypted]` | `tv`             | Samsung TV control          |
| `playwright`                   | `render`         | HTML -> PNG rendering       |
| `Pillow`                       | `palette`, `paintings` | Image processing       |
| `numpy`                        | `palette`                     | Numerical ops    |
| `tomllib` (stdlib)             | `config`         | TOML parsing                |

---

## Appendix A: Design Decisions Record

### A.1 LLM via CLI subprocess (not SDK)

**Decision**: Invoke LLM backends through their CLI tools (`gemini`, `codex`, `claude`) via
subprocess, rather than using Python SDKs directly.

**Rationale**: The user wants to switch between gemini CLI, codex CLI, and claude-code CLI
as interchangeable backends. Each CLI handles its own authentication, model routing, and
API versioning. Using subprocess means sfumato does not need to bundle API keys or SDK
dependencies for each provider.

**Trade-off**: Subprocess invocation is slower than direct API calls (~1-2s overhead per call)
and harder to capture structured errors. Accepted because LLM calls are infrequent (max
every 15 minutes for rotation, every 6h for curation) and the flexibility of swapping
backends outweighs the latency cost.

### A.2 Flat module structure (not packages)

**Decision**: All modules are flat `.py` files in `src/sfumato/`, not sub-packages.

**Rationale**: The system has ~14 modules, each with a clear single responsibility. Sub-packages
would add import complexity without meaningful organization benefit. If a module grows beyond
~500 lines, it can be split at that time.

**Trade-off**: If `paintings.py` needs separate files per art source, it may need to become a
sub-package later. Acceptable: YAGNI for now.

### A.3 State as JSON files (not SQLite)

**Decision**: Persist all state as JSON files, not SQLite.

**Rationale**: The state is small (hundreds of entries, not millions), rarely queried by
complex predicates, and needs to be human-readable for debugging. JSON files are trivially
inspectable, editable, and portable across Docker volumes.

**Trade-off**: No transactional writes, no concurrent access. Acceptable because sfumato
runs as a single daemon instance. Atomic writes via `os.replace()` prevent corruption.

### A.4 Dynamic batch sizing

**Decision**: News batch size (how many stories per rotation) is determined by the selected
painting's layout analysis (`recommended_stories`), not a fixed configuration value.

**Rationale**: A painting with a large quiet zone can fit 4-5 stories legibly. A painting
with limited space should show only 2-3. The LLM, having analyzed the composition, is best
positioned to recommend this.

**Trade-off**: Batch splitting becomes dynamic and may leave odd-sized remainders in the queue.
The queue handles this by allowing variable-size batches.

### A.5 LLM-based semantic matching (not categories or embeddings)

**Decision**: Use direct LLM matching to select the best painting for a given news tone.
Send painting descriptions + news tone to LLM and ask it to pick the best match.

**Rationale**: Fixed categories ("calm", "dramatic", "melancholic") are too coarse. The
free-form description approach captures nuances like "quiet melancholy vs. peaceful solitude"
that categorical labels cannot. Direct LLM matching is simpler than embedding + cosine
similarity and produces better results since the LLM can reason about the match holistically.

**Trade-off**: Requires one LLM text call per rotation (~15 min). Acceptable because the
call is fast and inexpensive (text-only, no vision). Cold-start: if fewer than 3 painting
descriptions are cached, falls back to random selection.
