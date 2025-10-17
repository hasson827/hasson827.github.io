#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-fix: add a single space around inline math $...$ in Markdown posts.
- Skips fenced code blocks (``` or ~~~)
- Skips inline code `...`
- Skips display math $$...$$ and \[...\]
- Only modifies single-dollar inline math
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POSTS = ROOT / "_posts"

# pattern for inline math that isn't $$...$$
INLINE_MATH = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$(?!\$)")

def patch_inline_math_spaces(text: str) -> str:
    lines = text.splitlines(True)  # keepends
    out_lines = []

    in_fence = False
    fence_marker = None  # ``` or ~~~
    in_display_dollars = False  # multi-line $$ ... $$ block

    for line in lines:
        # Detect fenced code blocks
        fence_match = re.match(r"^(\s*)(`{3,}|~{3,})(.*)$", line)
        if fence_match:
            marker = fence_match.group(2)
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker[:3] == (fence_marker or "```")[:3]:
                in_fence = False
                fence_marker = None
            out_lines.append(line)
            continue

        # Handle display math with $$ on separate lines (toggle when line contains only $$ ... $$ markers)
        # Start/end block when line has $$ and not escaped
        # We treat lines that begin with $$ or end with $$ as potential toggles
        if not in_fence:
            # Count unescaped $$ occurrences
            dollar_pairs = re.findall(r"(?<!\\)\$\$", line)
            # Toggle if odd count across the line (naive but works for typical blocks)
            if len(dollar_pairs) % 2 == 1:
                in_display_dollars = not in_display_dollars

        if in_fence or in_display_dollars:
            out_lines.append(line)
            continue

        # Now process inline code spans and only transform text outside them
        segments = []
        idx = 0
        for m in re.finditer(r"`+[^`]*`+", line):
            seg = line[idx:m.start()]
            segments.append((seg, True))   # True means process
            segments.append((line[m.start():m.end()], False))  # don't process code span
            idx = m.end()
        segments.append((line[idx:], True))

        new_line_parts = []
        for seg, do_process in segments:
            if not do_process:
                new_line_parts.append(seg)
                continue

            def repl(m: re.Match) -> str:
                s = m.group(0)
                start, end = m.start(), m.end()
                # Determine left/right neighbor within seg
                left_ws = (start == 0) or seg[start-1].isspace()
                right_ws = (end >= len(seg)) or seg[end:end+1].isspace()
                left = '' if left_ws else ' '
                right = '' if right_ws else ' '
                return f"{left}{s}{right}"

            seg = re.sub(INLINE_MATH, repl, seg)
            new_line_parts.append(seg)
        out_lines.append(''.join(new_line_parts))

    return ''.join(out_lines)


def main():
    changed = 0
    for md in sorted(POSTS.glob("*.md")):
        orig = md.read_text(encoding='utf-8')
        patched = patch_inline_math_spaces(orig)
        if patched != orig:
            md.write_text(patched, encoding='utf-8')
            changed += 1
            print(f"Patched: {md.relative_to(ROOT)}")
    print(f"Done. Files changed: {changed}")

if __name__ == "__main__":
    main()
