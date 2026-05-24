"""Generate the nemara-caregiver-app AppIcon-1024.png from the design spec.

The spec lives at module top so design tweaks are a single-line edit.
Re-run via `make icon` to regenerate after any change.

  Container — rounded square, radius 22.4% of side
  Background — #A04331 (terra)
  Glyph — lowercase "n", Fraunces variable font
    wght=500, opsz=144, SOFT=75
  Glyph color — #FBF8F0 (terra-ink)
  Glyph size — 69% of side
  Glyph offset — margin-top 4.5% of side, letter-spacing -1.6% of side,
    optically centered horizontally

We render against the Fraunces variable font (downloaded if not present)
instead of the bundled Fraunces144Soft-Medium.ttf static instance,
because static instances bake in axis values — the bundled "Soft"
variant locks SOFT at the maximum (~100), but the design spec wants
SOFT=75 specifically. The variable font lets us set it exactly via
Pillow's set_variation_by_axes.

Output:
    nemara-caregiver-app/NemaraCaregiver/Resources/Assets.xcassets/
        AppIcon.appiconset/AppIcon-1024.png
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── design spec ──────────────────────────────────────────────────
SIDE = 1024
RADIUS_PCT = 0.224
BACKGROUND = "#A04331"
GLYPH = "n"
GLYPH_COLOR = "#FBF8F0"
GLYPH_SIZE_PCT = 0.69
GLYPH_TOP_PCT = 0.225
LETTER_SPACING_PCT = -0.016
WEIGHT = 500
OPSZ = 144
SOFT = 75
WONK = 0

# ── paths ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
ICON_OUT = (
    REPO_ROOT.parent
    / "nemara-caregiver-app"
    / "NemaraCaregiver"
    / "Resources"
    / "Assets.xcassets"
    / "AppIcon.appiconset"
    / "AppIcon-1024.png"
)
FONT_LOCAL = Path("/tmp/Fraunces-VF.ttf")
FONT_URL = (
    "https://raw.githubusercontent.com/undercasetype/Fraunces/master/"
    "fonts/variable/Fraunces%5BSOFT%2CWONK%2Copsz%2Cwght%5D.ttf"
)


def _ensure_font() -> Path:
    if FONT_LOCAL.exists() and FONT_LOCAL.stat().st_size > 100_000:
        return FONT_LOCAL
    print(f"downloading Fraunces variable font → {FONT_LOCAL}")
    urllib.request.urlretrieve(FONT_URL, FONT_LOCAL)
    return FONT_LOCAL


def main() -> int:
    font_path = _ensure_font()

    img = Image.new("RGBA", (SIDE, SIDE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded square background
    radius = round(RADIUS_PCT * SIDE)
    draw.rounded_rectangle(
        (0, 0, SIDE - 1, SIDE - 1), radius=radius, fill=BACKGROUND,
    )

    # Load the variable font and apply the design's axis values.
    # `set_variation_by_axes` takes values in the font's fvar order —
    # we look up the tags so we don't hard-code an assumption about
    # the order in the .ttf.
    glyph_size = round(GLYPH_SIZE_PCT * SIDE)
    font = ImageFont.truetype(str(font_path), glyph_size)

    # Pillow returns axes as dicts keyed by display name; map by name
    # so order changes in the .ttf can't silently swap values.
    axes_meta = font.get_variation_axes()
    name_to_value = {
        b"Optical Size": OPSZ,
        b"Softness": SOFT,
        b"Weight": WEIGHT,
        b"Wonky": WONK,
    }
    ordered = [name_to_value[a["name"]] for a in axes_meta]
    print(
        f"font axes (fvar order): {[a['name'].decode() for a in axes_meta]} "
        f"→ values: {ordered}",
    )
    font.set_variation_by_axes(ordered)

    # Compute glyph position.
    # `textbbox` returns the inked bounding box of "n" at this size.
    # The "n" glyph has no descender; its bbox top is the x-height edge.
    bbox = draw.textbbox((0, 0), GLYPH, font=font)
    text_w = bbox[2] - bbox[0]

    # Horizontal: optically centered (geometric center, then letter-
    # spacing pushes it slightly). For a single glyph the spec's
    # letter-spacing is effectively a half-application on each side.
    x = (SIDE - text_w) // 2 - bbox[0] + round(LETTER_SPACING_PCT * SIDE / 2)

    # Vertical: "margin-top 4.5% of side · optically centered" means
    # the em box's top (≈ cap-height line) sits at 4.5% from the
    # container's top, not the inked glyph's top. For lowercase "n"
    # the glyph itself starts lower (at x-height), so this lands the
    # letter visually in the upper-mid region. Pillow's draw origin
    # is already the em-box top, so no bbox subtraction.
    y = round(GLYPH_TOP_PCT * SIDE)

    draw.text((x, y), GLYPH, font=font, fill=GLYPH_COLOR)

    ICON_OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(ICON_OUT, "PNG")
    print(f"✓ wrote {ICON_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
