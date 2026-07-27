# Mixed-dataset joint-training rewrite

This is the clean handoff package for the separate July 26, 2026 rewrite.
It is an experimental post-submission variant, not the final PNAS paper.

## Review first

- `Mixed_Dataset_Joint_Training_Main_20260726.pdf`
- `Mixed_Dataset_Joint_Training_Supplement_20260726.pdf`
- `REWRITE_NOTES.md`

The manuscript now presents the primary NAFNet as a single model trained
from random initialization with deterministic interleaving of 247,150
pixel-registered fog-chamber updates and 2,600 synthetic-fog updates.
The 5x-synthetic schedule is retained only as a sensitivity check.

## Clean LaTeX sources

The self-contained source folders are:

- `latex_main/`
- `latex_supplement/`

Build the main paper:

```bash
cd latex_main
pdflatex -interaction=nonstopmode -halt-on-error main_text.tex
bibtex main_text
pdflatex -interaction=nonstopmode -halt-on-error main_text.tex
pdflatex -interaction=nonstopmode -halt-on-error main_text.tex
```

Build the supplement:

```bash
cd latex_supplement
pdflatex -interaction=nonstopmode -halt-on-error defogging_supplement.tex
bibtex defogging_supplement
pdflatex -interaction=nonstopmode -halt-on-error defogging_supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error defogging_supplement.tex
```

Both folders were independently rebuilt from clean temporary copies.
The expected outputs are 13 and 16 letter-size pages, respectively.

## Other deliverables

- `evaluation_summary/`: the full common-evaluation report and CSV tables
- `SHA256SUMS.txt`: checksums for every packaged file except itself

The public-dataset evaluation is retained as a direct-transfer diagnostic,
not promoted to a new headline benchmark in the manuscript.

The supplementary videos remain in the local research archive and are not
duplicated in this GitHub experiment directory.

## Post-freeze result

The 0.2x-synthetic run completed after this paper package was frozen. Its
config and summary are released under `../results/synthetic_0p2x/`, but the
paper PDF remains the audited July 26 version.
