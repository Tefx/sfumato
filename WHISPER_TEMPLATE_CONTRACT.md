# Whisper Template Contract

## Scope

This document defines the non-rendering contract for production whisper text placeholders in `templates/painting_text.html`, `templates/magazine.html`, and `templates/portrait.html`.

## Required Variables

All three templates MUST expose these placeholder variables using existing `{{PLACEHOLDER}}` syntax:

| Variable | Type | Contract |
| --- | --- | --- |
| `WHISPER_POSITION` | CSS declaration fragment | Absolute-position token supplied by layout analysis. It describes the reserved whisper zone only and MUST remain compatible with the template's existing positioning model. |
| `WHISPER_COLOR` | Color token | Final text color token for whisper copy. It MUST be visually subordinate to primary news text while remaining readable on a 4K frame. |
| `WHISPER_SHADOW` | CSS text-shadow token | Shadow token used to preserve readability over art backgrounds or dark panels. |
| `WHISPER_TEXT` | Escaped string | Whisper copy payload. It is the only content node inside the contract placeholder. |

## Shared Placeholder Shape

Each template MUST include the same inert placeholder artifact:

```html
<template id="whisper-contract">
  <div
    class="whisper-contract"
    data-whisper-position="{{WHISPER_POSITION}}"
    data-whisper-color="{{WHISPER_COLOR}}"
    data-whisper-shadow="{{WHISPER_SHADOW}}"
  >{{WHISPER_TEXT}}</div>
</template>
```

Contract notes:
- The `template` element is intentionally non-rendering; it defines structure without implementing final behavior.
- The root node class and `data-*` attributes MUST remain identical across all three templates.
- `WHISPER_TEXT` MUST remain the direct text payload of the placeholder root so downstream materialization logic can treat all templates uniformly.

## Shared Typography Constraints

- Whisper copy is secondary editorial text, not a headline or body block.
- Whisper typography MUST remain visually below primary news content in hierarchy.
- Whisper readability target is tuned for a 3840x2160 output frame; prototype evidence in `PROTOTYPING.md#11` sets a nominal readable floor at 28px with shadow support.
- The contract therefore REQUIRES both color and shadow tokens to remain externally supplied rather than hard-coded in template markup.

## Shared Positioning Constraints

- Whisper placement MUST be planned together with the news text zone; the two zones are mutually exclusive.
- Whisper placement SHOULD prefer the opposite visual quadrant from the main news block, following `PROTOTYPING.md#11`.
- `WHISPER_POSITION` MUST describe placement only; it must not rename, replace, or invalidate existing template placeholders such as `TEXT_POSITION`, `TEXT_WIDTH`, `SCRIM_POSITION`, `LEFT_WIDTH`, or `PAINTING_WIDTH`.
- The placeholder artifact MUST stay outside repeated story/news item loops so a single whisper zone exists per frame.

## Compatibility Expectations

- This contract is additive: existing template variables remain unchanged and keep their current meaning.
- Existing renderers that substitute `{{PLACEHOLDER}}` values can continue to process the templates without new syntax.
- Future implementation work may materialize the inert placeholder into visible markup or CSS, but that work is out of scope for this contract.
