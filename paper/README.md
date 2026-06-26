# Paper Files

This folder contains the paper PDFs and clean LaTeX source folders used for the public release.

## PDFs

- `main_text.pdf`: main manuscript.
- `defogging_supplement.pdf`: supplemental document.

## LaTeX Sources

- `main_latex/`: source files, figure images, bibliography, and Optica style files for rebuilding the main manuscript.
- `supplement_latex/`: source files, supplement figures, bibliography, tables, and style files for rebuilding the supplemental document.

Build from inside each source folder:

```bash
pdflatex -interaction=nonstopmode main_text.tex
bibtex main_text
pdflatex -interaction=nonstopmode main_text.tex
pdflatex -interaction=nonstopmode main_text.tex
```

For the supplement, replace `main_text` with `defogging_supplement`.

The source folders intentionally exclude local backups and transient LaTeX build files such as `.aux`, `.log`, `.blg`, `.out`, and generated PDFs.
