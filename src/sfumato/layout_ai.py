"""Use Claude vision to analyze a painting and design optimal text layout."""

from __future__ import annotations

import anthropic
import base64
import json
from pathlib import Path


LAYOUT_PROMPT = """\
你是一个视觉排版设计师。我需要在这幅画作上叠加新闻文字，用于Samsung The Frame电视显示。

电视分辨率: 3840x2160px
画作会全屏显示，文字直接排在画面上。

请分析这幅画的构图，告诉我：

1. **文字应该放在哪里** — 找到画面中视觉密度最低的区域（天空、水面、雾气、暗角、纯色背景等）
2. **文字颜色** — 根据该区域的明暗决定用浅色字还是深色字
3. **是否portrait** — 判断这幅画是横幅还是竖幅构图

输出严格的JSON格式：

```json
{
  "orientation": "landscape" 或 "portrait",
  "painting_title": "画作名称（如果能识别）",
  "painting_artist": "画家（如果能识别）",
  "text_zone": {
    "position": "top-left" / "top-right" / "bottom-left" / "bottom-right" / "left-side" / "right-side",
    "reason": "简述为什么选这个区域"
  },
  "colors": {
    "text_primary": "#xxxxxx",
    "text_secondary": "#xxxxxx",
    "text_dim": "#xxxxxx",
    "text_shadow": "CSS text-shadow value",
    "scrim_color": "rgba(x,x,x,0.x) — 用于轻微压暗/提亮文字区域的颜色"
  },
  "css": {
    "text_position": "CSS position properties (e.g. 'top: 100px; right: 160px;')",
    "text_max_width": "e.g. '1500px'",
    "scrim_position": "CSS position properties for the scrim div",
    "scrim_size": "CSS width/height for scrim",
    "scrim_gradient": "CSS radial-gradient value"
  },
  "portrait_layout": null 或 {
    "painting_width_percent": 45-55,
    "left_panel_color": "#xxxxxx",
    "right_panel_color": "#xxxxxx",
    "info_side": "left" / "right" / "both"
  }
}
```

注意：
- text_position用CSS absolute定位，距离边缘至少120px
- text_max_width不超过画面宽度的45%（landscape时）
- scrim是一个柔和的渐变，不是方框，用radial-gradient
- portrait画作需要给出portrait_layout，画作居中显示，两侧面板放文字
- 颜色要和画面协调，不突兀

只输出JSON，不要其他文字。
"""


def analyze_painting(image_path: Path) -> dict:
    """Send painting to Claude Vision and get layout parameters."""
    client = anthropic.Anthropic()

    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    suffix = image_path.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(suffix, "image/jpeg")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": LAYOUT_PROMPT,
                    },
                ],
            }
        ],
    )

    raw = message.content[0].text
    # Strip markdown code fences if present
    if "```" in raw:
        raw = raw.split("```json")[-1].split("```")[0] if "```json" in raw else raw.split("```")[1].split("```")[0]

    return json.loads(raw)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m frame_terminal.layout_ai <image_path>")
        sys.exit(1)

    result = analyze_painting(Path(sys.argv[1]))
    print(json.dumps(result, indent=2, ensure_ascii=False))
