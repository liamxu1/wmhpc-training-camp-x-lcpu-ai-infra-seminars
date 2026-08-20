#!/usr/bin/env bash
# handout 构建:md 源 → tex → pdf。
# 用法:./build.sh [src/assignment02.md]
# 依赖:pandoc(>=3)、xelatex、Noto CJK 字体。
set -e
cd "$(dirname "$0")"
SRC=${1:-src/assignment02.md}
BASE=$(basename "$SRC" .md)
pandoc "$SRC" \
    --from markdown+fenced_divs+pipe_tables+raw_tex \
    --template template.tex \
    --lua-filter filters/boxes.lua \
    --syntax-highlighting=idiomatic \
    -o "$BASE.tex"
xelatex -interaction=nonstopmode "$BASE.tex" >/dev/null || true
xelatex -interaction=nonstopmode "$BASE.tex" | tail -3
echo "生成 $BASE.pdf"
