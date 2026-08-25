#!/usr/bin/env python3
"""Render privacy-notice.md into the surfaces that display it.

The notice used to be written out by hand in four places and had already
drifted apart by the time anyone noticed (2026-08-23: the iOS screen and the
consent-gate PDF were describing different retention rules). This script makes
the markdown the only place it is written.

    python3 docs/legal/render_privacy.py --swift <PrivacyPolicyView.swift>

PDF generation is separate because it needs pandoc + the AppKit helper:

    pandoc privacy-notice.md -t html --standalone -o notice.html
    ./html2pdf notice.html privacy.pdf
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "privacy-notice.md"


def parse(md: str) -> tuple[str, list[tuple[str, list]]]:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    effective = ""
    m = re.search(r"^Effective Date:\s*(.+)$", md, re.M)
    if m:
        effective = m.group(1).strip()

    sections: list[tuple[str, list]] = []
    title: str | None = None
    body: list = []
    para: list[str] = []
    bullets: list[str] = []

    def flush_para() -> None:
        if para:
            body.append(("paragraph", " ".join(para).strip()))
            para.clear()

    def flush_bullets() -> None:
        if bullets:
            body.append(("bullets", list(bullets)))
            bullets.clear()

    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("## "):
            flush_para(); flush_bullets()
            if title is not None:
                sections.append((title, body))
            title, body = line[3:].strip(), []
        elif line.startswith("### "):
            flush_para(); flush_bullets()
            body.append(("subheading", line[4:].strip()))
        elif line.startswith("- "):
            flush_para()
            bullets.append(line[2:].strip())
        elif not line:
            flush_para(); flush_bullets()
        elif title is not None:
            flush_bullets()
            para.append(line)

    flush_para(); flush_bullets()
    if title is not None:
        sections.append((title, body))
    return effective, sections


def swift_string(s: str) -> str:
    # Swift string literal: escape backslash and quote only. The view feeds
    # these to AttributedString(markdown:), so ** and * stay as-is.
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def to_swift(sections: list[tuple[str, list]]) -> str:
    out = ["    static let sections: [PrivacySection] = ["]
    for title, body in sections:
        out.append(f"        PrivacySection(title: {swift_string(title)}, body: [")
        for kind, val in body:
            if kind == "bullets":
                out.append("            .bullets([")
                for b in val:
                    out.append(f"                {swift_string(b)},")
                out.append("            ]),")
            else:
                out.append(f"            .{kind}({swift_string(val)}),")
        out.append("        ]),")
    out.append("    ]")
    return "\n".join(out)


def jsx_text(s: str) -> str:
    """Escape for JSX text, then re-introduce **bold** as <strong>.

    Straight double quotes become curly pairs rather than &quot;: react's
    no-unescaped-entities rule rejects the raw character, and the quoted
    sample-signage text in the notice reads better with real quotation marks.
    """
    s = (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("{", "&#123;")
        .replace("}", "&#125;")
        .replace("'", "&rsquo;")
    )
    parts = s.split('"')
    if len(parts) > 1:
        s = parts[0]
        for i, part in enumerate(parts[1:]):
            s += ("&ldquo;" if i % 2 == 0 else "&rdquo;") + part
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)


def to_tsx(effective: str, sections: list[tuple[str, list]]) -> str:
    out = [
        "// GENERATED from nemara-backend/docs/legal/privacy-notice.md.",
        "// Do not hand-edit: edit the markdown and re-run render_privacy.py.",
        "import Link from 'next/link';",
        "",
        "export const metadata = {",
        "  title: 'Privacy Notice — Memoria',",
        "  description:",
        "    'How Memoria collects, uses, discloses, and protects personal "
        "information; your rights; and the choices available to you.',",
        "};",
        "",
        "export default function PrivacyPage() {",
        "  return (",
        '    <main className="mx-auto max-w-2xl px-6 py-12 text-ink">',
        '      <nav className="mb-8 text-sm">',
        '        <Link href="/login" className="text-terra underline">',
        "          ← Back to Memoria",
        "        </Link>",
        "      </nav>",
        "",
        '      <h1 className="font-serif text-4xl tracking-tight">Privacy Notice</h1>',
        f'      <p className="mt-2 text-sm text-ink3">Effective Date: {effective}</p>',
        "",
    ]
    for title, body in sections:
        out.append(
            f'      <h2 className="mt-10 font-serif text-2xl">{jsx_text(title)}</h2>'
        )
        for kind, val in body:
            if kind == "bullets":
                out.append(
                    '      <ul className="mt-3 list-disc pl-6 text-ink2 space-y-1">'
                )
                for b in val:
                    out.append(f"        <li>{jsx_text(b)}</li>")
                out.append("      </ul>")
            elif kind == "subheading":
                out.append(
                    '      <h3 className="mt-6 font-semibold text-ink">'
                    f"{jsx_text(val)}</h3>"
                )
            else:
                out.append(
                    f'      <p className="mt-3 text-ink2">{jsx_text(val)}</p>'
                )
        out.append("")
    out += ["    </main>", "  );", "}", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swift", type=Path, required=True)
    ap.add_argument("--tsx", type=Path)
    args = ap.parse_args()

    effective, sections = parse(SOURCE.read_text(encoding="utf-8"))

    if args.tsx:
        args.tsx.write_text(to_tsx(effective, sections), encoding="utf-8")
        print(f"{args.tsx.name}: {len(sections)} sections")

    swift = args.swift.read_text(encoding="utf-8")

    new_block = to_swift(sections)
    swift, n = re.subn(
        r"    static let sections: \[PrivacySection\] = \[.*?\n    \]",
        lambda _m: new_block,
        swift,
        flags=re.S,
    )
    if n != 1:
        raise SystemExit(f"expected 1 sections block, replaced {n}")

    stamp = f"EFFECTIVE {effective.upper()}"
    swift, n = re.subn(r'Text\("EFFECTIVE [^"]*"\)', f'Text("{stamp}")', swift)
    if n != 1:
        raise SystemExit(f"expected 1 effective-date stamp, replaced {n}")

    args.swift.write_text(swift, encoding="utf-8")
    print(f"{args.swift.name}: {len(sections)} sections, stamp {stamp!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
