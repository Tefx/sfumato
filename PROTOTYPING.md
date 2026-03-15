# Prototyping Notes & Validated Findings

This document captures all conclusions from the prototype exploration phase.
Implementation agents should read this before starting any step.

---

## 1. Samsung The Frame TV API

### Verified Working (2024 Frame, Tizen 8, IP 192.168.88.22)

Despite community reports that Samsung removed WebSocket Art Mode API from Tizen 6.5+,
**our 2024 model fully supports it**.

```python
from samsungtvws import SamsungTVWS

tv = SamsungTVWS(host='192.168.88.22', port=8002, name='FrameTerminal')
art = tv.art()

# All of these work:
art.supported()          # → True
art.available()          # → list of art items (83 items found)
art.get_artmode()        # → "on" / "off"
art.get_current()        # → {"content_id": "MY_F0003", ...}
art.get_device_info()    # → {"resolution_type": "UHD", "support_motion_sensor": "FALSE", ...}
art.get_api_version()    # → "4.3.4.0"

# Upload and display:
art.upload(data, file_type='PNG', matte='none')
art.select_image('MY_F0003', show=True)

# Note: select_image works but TV screen must be on/art mode active to see change
# If TV is in standby, the image is selected but screen stays dark
```

### Key Findings
- **First connection** requires accepting a pairing prompt on the TV
- **Port 8002** (SSL) works, port 8001 may also work
- **Content IDs** for user uploads follow pattern `MY_F0001`, `MY_F0002`, etc.
- **Category** for user photos is `MY-C0002`
- **Image date** is populated on upload: `"2026:03:14 17:24:47"`
- `set_artmode(True)` can hang/timeout — avoid calling it; just use `select_image`
- `get_thumbnail()` hangs indefinitely on 2024 model — **do not use**
- `support_motion_sensor` is FALSE on this model — screen won't auto-wake on proximity

### API Methods Available
```
available, change_matte, delete, delete_list, get_api_version, get_artmode,
get_artmode_settings, get_auto_rotation_status, get_brightness, get_color_temperature,
get_current, get_device_info, get_matte_list, get_photo_filter_list, get_rotation,
get_slideshow_status, get_thumbnail, get_thumbnail_list, select_image, set_artmode,
set_auto_rotation_status, set_brightness, set_brightness_sensor_setting,
set_color_temperature, set_favourite, set_motion_sensitivity, set_motion_timer,
set_photo_filter, set_slideshow_status, supported, upload
```

---

## 2. Rendering: Playwright vs PIL

### Conclusion: **Use Playwright. PIL is not acceptable.**

Tested both on the actual TV. PIL text rendering is visibly inferior at TV viewing distance.

| Aspect | Playwright | PIL |
|--------|-----------|-----|
| Font rendering | Browser-grade antialiasing, excellent | Rough, visible jaggedness on TV |
| Gradients/scrim | CSS radial-gradient, smooth | Manual ellipse overlay, banded |
| Text layout | CSS auto-wrap, line-height, letter-spacing | Manual character-by-character wrapping |
| Font selection | Google Fonts (online load) | System fonts only (STHeiti on macOS) |
| Chinese text | Perfect with Noto Sans SC / Noto Serif SC | Acceptable with STHeiti but no weight control |
| Rendering speed | ~2-3s (browser startup) | ~0.3s |
| Docker size | +400MB (chromium) | minimal |
| **TV verdict** | **Excellent** | **Rejected by user** |

### Playwright Setup
```python
from playwright.async_api import async_playwright

async with async_playwright() as p:
    browser = await p.chromium.launch()
    page = await browser.new_page(viewport={"width": 3840, "height": 2160})
    await page.set_content(html, wait_until="networkidle")
    await page.wait_for_timeout(1000)  # let Google Fonts load
    await page.screenshot(path=str(output_path), type="png")
    await browser.close()
```

### Google Fonts Used
```css
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&family=Noto+Serif+SC:wght@400;600;700;900&family=JetBrains+Mono:wght@300;400&family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=Inter:wght@300;400;500;600&display=swap');
```

---

## 3. Layout Templates Tested

### 7 templates created, 2 recommended for production:

| Template | File | Use Case | Status |
|----------|------|----------|--------|
| `painting_text.html` | Landscape, text in quiet zone | **Primary — recommended** |
| `portrait.html` | Portrait painting, 3-column layout | **Recommended for portrait** |
| `magazine.html` | 72/28 split, painting left + text right | **Fallback for dense compositions** |
| `art_overlay.html` | Frosted glass cards over painting | Rejected by user — obscures art |
| `art_minimal.html` | Bottom strip banner | Rejected — looks like TV news ticker |
| `gallery_wall.html` | Painting framed on wall, news as labels | Rejected — wastes Frame TV's bezel |
| `news_poster.html` | Pure dark newspaper layout, no painting | Not preferred — no art |

### painting_text.html — Primary Template

Key design principle: **The Frame TV bezel IS the frame. Painting fills 100% of screen.**

Template variables:
```
{{BG_IMAGE}}         — file:// URI to painting
{{SCRIM_GRADIENT}}   — CSS gradient applied as text-zone background (auto-sizes with content)
{{TEXT_POSITION}}     — CSS absolute position for text zone
{{TEXT_WIDTH}}        — max-width of text zone
{{TEXT_COLOR}}        — primary text color (titles)
{{TEXT_COLOR_SEC}}    — secondary text color (body)
{{TEXT_COLOR_DIM}}    — dim text color (dateline)
{{TEXT_SHADOW}}       — CSS text-shadow
{{DATELINE}}          — date + source line
{{NEWS_BLOCKS}}       — HTML blocks for stories
```

Each news block:
```html
<div class="news-block">
  <div class="title">标题</div>
  <div class="body">正文摘要</div>
</div>
```

### portrait.html — For Portrait Paintings

Layout: left panel (date/artist info) + center painting + right panel (news)
- Painting displayed at full height, natural aspect ratio
- Panel background color extracted from painting edge pixels
- Example: Friedrich "Wanderer" → dark blue-grey panels (#1c1e20)

### Font Sizes (for 4K, 2-3m viewing distance)
```
Title:    48-56px, font-weight 700
Body:     28-32px, font-weight 300, line-height 1.7
Dateline: 22-26px, monospace, letter-spacing 4-5px
```

---

## 4. LLM CLI for Layout Analysis

### All three CLIs work for painting analysis:

#### Gemini CLI
```bash
gemini -p "分析这幅画的构图..." -y --sandbox false
# Reads files via tool calls
# Returns structured JSON
# Good: concise output, fast
```

#### Codex CLI
```bash
codex exec --full-auto "分析画作构图..."
# Requires git repo (run git init first)
# Views images via tool
# Good: structured CSS output, includes width percentages
```

#### Claude Code
```
# Direct analysis in conversation
# No CLI subprocess needed — IS the current agent
# Good: most nuanced color choices, matched warm tones to Hokusai
```

### Comparison on Hokusai "Great Wave"

| | Gemini | Codex | Claude Code |
|---|---|---|---|
| Text zone | top-right | top-right (10%, 8%) | top-right (120px, 140px) |
| Text color | #1d2d50 (deep blue) | #1F355E (deep blue) | #2a1f14 (deep brown) |
| Scrim | White radial, subtle | Beige linear, aggressive | Warm elliptical, minimal |
| Paint. impact | Medium | Highest (large area lightened) | Lowest (most respectful of painting) |
| Color harmony | Good | Good | Best (warm brown matches Hokusai palette) |

### Key Learnings
- All three identify the same quiet zone (top-right sky for Hokusai)
- Color choice matters: Codex/Gemini default to blue text; Claude chose brown to match the warm ukiyo-e palette
- Scrim aggressiveness varies: Codex most aggressive (better readability), Claude most subtle (better aesthetics)
- For production: let the LLM choose, but provide guidance in prompt about respecting the painting

### Prompt Structure for Layout Analysis
The LLM should output JSON with these fields:
```json
{
  "orientation": "landscape|portrait",
  "painting_title": "...",
  "painting_artist": "...",
  "text_zone": {"position": "top-right", "reason": "..."},
  "recommended_story_count": 3,
  "colors": {
    "text_primary": "#hex",
    "text_secondary": "#hex",
    "text_dim": "#hex",
    "text_shadow": "CSS text-shadow value",
    "scrim_color": "rgba(...)"
  },
  "css": {
    "text_position": "CSS absolute positioning",
    "text_max_width": "px value",
    "scrim_position": "CSS positioning",
    "scrim_size": "CSS width/height",
    "scrim_gradient": "CSS gradient value"
  },
  "description": "free-form mood/atmosphere description for semantic matching",
  "portrait_layout": null | { ... }
}
```

---

## 5. Painting Sources

### Tested Sources

| Source | Status | Issues |
|--------|--------|--------|
| Wikimedia Commons | **Rate-limited** (429 after 2-3 requests) | Need browser-like User-Agent + delays between requests |
| Art Institute of Chicago (IIIF) | **Cloudflare blocked** (403) | Cannot use programmatically |
| Met Museum API | **JSON API works**, image download blocked | API for metadata OK, images behind Cloudflare |

### Wikimedia Workaround
```python
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
}
# Add 2-3 second delays between requests
# Use thumbnail URLs (e.g., /thumb/.../3840px-...) not /full/
# Rate limit: ~1 request per 3 seconds to avoid 429
```

### Paintings Successfully Downloaded and Tested
1. **Van Gogh — Starry Night** (starry_night_4k.jpg) — dark, swirling, energetic
2. **Monet — Water Lilies 1906** (monet_bg_4k.jpg) — soft, contemplative, blue-green
3. **Hokusai — Great Wave** (hokusai_bg_4k.jpg) — dramatic, tense, warm/cool contrast
4. **Friedrich — Wanderer above the Sea of Fog** (friedrich_raw.jpg) — contemplative, foggy, portrait orientation

### Image Preparation Pipeline
```python
from PIL import Image

img = Image.open(raw_path)
w, h = img.size
target_ratio = 3840 / 2160

# Crop to 16:9
if w/h > target_ratio:  # too wide
    new_w = int(h * target_ratio)
    left = (w - new_w) // 2
    img = img.crop((left, 0, left + new_w, h))
else:  # too tall
    new_h = int(w / target_ratio)
    top = (h - new_h) // 2
    img = img.crop((0, top, w, top + new_h))

img = img.resize((3840, 2160), Image.LANCZOS)
```

---

## 6. Palette Extraction

```python
from PIL import Image
import numpy as np

img = Image.open(path)
arr = np.array(img.convert('L'))  # grayscale

# Grid analysis for finding quiet zones
grid_h, grid_w = 6, 8
cell_h, cell_w = arr.shape[0] // grid_h, arr.shape[1] // grid_w
for r in range(grid_h):
    for c in range(grid_w):
        cell = arr[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
        variance = np.var(cell)    # low = flat area, good for text
        brightness = np.mean(cell)  # determines light vs dark text

# Edge color extraction (for portrait panel backgrounds)
left_edge = np.array(img)[:, :20, :].mean(axis=(0,1))
right_edge = np.array(img)[:, -20:, :].mean(axis=(0,1))
# Darken by 0.2x for panel background
```

**Note:** PIL/numpy analysis is useful for orientation detection and palette extraction,
but NOT sufficient for layout decisions. LLM vision analysis is required for understanding
composition and choosing text placement. See Section 4.

---

## 7. News Fetching

### TLDR Parsing
TLDR RSS (`https://tldr.tech/api/rss/tech`) returns entries with titles but empty summaries.
Must fetch the actual page and parse HTML:

```python
# Fetch page: https://tldr.tech/tech/2026-03-14
# Split by <h3> tags
# Section headers: "Big Tech & Startups", "Science & Futuristic Technology", etc.
# Articles have "(X minute read)" in title
# Summary is next text block after </h3> with length > 40 chars
```

### News Curation — Gemini CLI Validated
Successfully used `gemini -p` to:
1. Take 20 raw articles
2. Group into 3 batches by theme/mood
3. Translate to Chinese
4. Generate 50-80 word summaries
5. Tag each batch with mood + themes

Output structure validated:
```json
[
  {
    "batch_id": 1,
    "mood": "energetic",
    "themes": ["ai-tools", "developer-platforms"],
    "stories": [
      {"headline": "中文标题", "summary": "50-80字中文摘要"}
    ]
  }
]
```

---

## 8. Semantic Mood Matching — Validated

### Concept
Match news batch emotional tone to painting atmosphere using free-form descriptions + LLM-based matching.
(Note: originally used embeddings + cosine similarity, replaced with direct LLM matching for simplicity.)

### Test Results (3 batches matched to 3 paintings)

| News Batch | Mood | Matched Painting | Result |
|------------|------|-----------------|--------|
| AI tools explosion (Claude viz, Gemini embed, Perplexity) | energetic | Starry Night (Van Gogh) | ✓ Swirling energy matches tech excitement |
| Big tech acquisitions (Meta+Moltbook, Musk Macrohard) | tense | Great Wave (Hokusai) | ✓ Dramatic tension matches corporate warfare |
| Societal reflection (AI vs jobs, ATM vs iPhone) | contemplative | Wanderer in Fog (Friedrich) | ✓ Solitary contemplation matches deep thinking |

All three were uploaded to the TV and confirmed working. User approved the concept.

### Implementation Path
- Each painting: LLM generates free-form description → cache
- Each news batch: LLM generates tone description (part of curation prompt)
- Selection: LLM compares tone with painting descriptions and picks best match
- No fixed mood categories needed

---

## 9. Rejected Approaches

1. **PIL rendering** — Text quality unacceptable on TV at viewing distance
2. **Gallery wall layout** — Wastes Frame TV's natural bezel as frame
3. **Frosted glass overlay** — Obscures painting too much
4. **Bottom ticker banner** — Looks like cheap TV news
5. **Fixed mood categories** — Too rigid; free-form descriptions + LLM matching are better
6. **Downloading art from TV** — `get_thumbnail()` hangs on 2024 model
7. **set_artmode()** — Can hang; use select_image() instead

---

## 10. Reference Files in output/

These files were generated during prototyping and can be used as visual references:

- `poster_*.png` — Early pure-newspaper style renders
- `art_poster.png` — Starry Night + frosted glass overlay (rejected)
- `art_poster_cn.png` — Monet + frosted glass Chinese (rejected)
- `hokusai_poster.png` — Hokusai + bottom banner (rejected)
- `gallery_wall.png` — Gallery wall style (rejected)
- `painting_text_starry.png` — Starry Night + text in quiet zone v1
- `painting_text_v2.png` — Same with improved scrim (accepted)
- `painting_text_friedrich.png` — Friedrich + dark text on fog (accepted)
- `portrait_friedrich.png` — Portrait 3-column layout (accepted)
- `magazine_hokusai.png` — 72/28 magazine split (accepted)
- `hokusai_gemini.png` — Gemini CLI layout analysis result
- `hokusai_codex.png` — Codex CLI layout analysis result
- `hokusai_claude.png` — Claude Code layout analysis result
- `mood_energetic.png` — Starry Night + energetic news batch
- `mood_tense.png` — Hokusai + tense news batch
- `mood_contemplative.png` — Friedrich + contemplative news batch
- `pil_hokusai.png` — PIL rendering (rejected, poor quality)

---

## 11. Whisper Text (Art Facts) — 验证结论

### 视觉效果测试

| 参数 | 结果 |
|------|------|
| rgba 0.25 透明度 | 太淡，电视上不可见 |
| rgba 0.45 透明度 | 仍然太淡 |
| rgba 0.75 + 24px | 可见但颜色不对 |
| 亮度自适应 + 28px + shadow | 效果可接受 |

### 关键发现
- **whisper text 和新闻文字可能重叠** — 两者都不知道对方的位置
- **必须在 LLM layout analysis 时同时规划 text_zone 和 whisper_zone，保证不重叠**
- 颜色和位置都需要 LLM 根据画面决定（和新闻文字同理）
- 字号 28px、带 text-shadow 在正常观看距离可读
- 不需要追求"走近才看见"——电视是偶尔瞄一眼的场景

### 设计原则
- whisper 位置在新闻的对角（新闻右上 → whisper 左下）
- LLM 一次调用输出两个互斥区域
- 视觉层级：新闻 > whisper（字号更小、颜色更淡、但仍可读）

## 12. Watch Daemon Test Results (2026-03-15)

### Test 1-4: PASS
- First rotation: ✓ renders produced
- Second rotation: ✓ new render, replay queue grows
- Forced refresh: ✓ clear queue → re-curate 20 new batches
- Backfill: pool 29/200, should trigger on idle

### Test 5: FAIL — SIGTERM
- Daemon didn't exit within 5s of SIGTERM, required SIGKILL
- Health file not generated
- Issue: signal handler may not interrupt asyncio event loop

### Layout Quality Issues Found
- Dense paintings (Constable forest, Ricci Baptism): text placed over busy areas
- 4 stories with long summaries overflow from corner into painting center
- LLM recommended_stories sometimes too high for available quiet space
- Need: stronger variance→magazine routing in prompt (added in commit f0b3086)

### Watch Daemon SIGTERM Retest (2026-03-15)

- SIGTERM during sleep phase: ✅ PASS — exits within 5s, health file written
- SIGTERM during LLM call: ⚠️ blocked until current call completes (~10-60s)
- Docker stop_grace_period: 90s (sufficient)
- Health file: ✅ last_action.json correctly written on shutdown
