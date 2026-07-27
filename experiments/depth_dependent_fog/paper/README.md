# Depth-extension paper variant

These are experimental paper artifacts, not the final PNAS submission.

- `depth_extension_main_20260723.pdf`
- `depth_extension_supplement_20260723.pdf`
- `source/main/`
- `source/supplement/`

The conservative variant keeps randomized synthetic fog as the primary
workflow and presents ZoeDepth-dependent fog as an extension. During release
preparation, the cross-generator prose was corrected to match the underlying
30-model table: depth-dependent checkpoints were `-2.08 dB` relative to
randomized checkpoints on randomized fog and `+3.41 dB` on depth fog.

Build the main paper:

```bash
cd source/main
pdflatex -interaction=nonstopmode -halt-on-error main_text.tex
bibtex main_text
pdflatex -interaction=nonstopmode -halt-on-error main_text.tex
pdflatex -interaction=nonstopmode -halt-on-error main_text.tex
```

Build the supplement with the same sequence using
`defogging_supplement.tex`.
