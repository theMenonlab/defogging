# Mixed-Dataset Joint-Training Rewrite

## Scope

- Baseline: the clean July 25, 2026 main-paper and supplement LaTeX packages.
- Output: this directory only. The July 25 final review package is unchanged.
- Purpose: rewrite the manuscript around simultaneous, deterministic interleaving of fog-chamber and synthetic Mapillary updates, instead of fog-chamber pretraining followed by synthetic fine-tuning.

## Primary model and training evidence

The primary mixed model is the completed matched-budget run:

- Initialization: random NAFNet weights; no fog-chamber checkpoint.
- Chamber data: 4,943 paired training images and 552 held-out pairs.
- Synthetic data: 18,000 / 2,000 / 5,000 Mapillary train / validation / test images.
- Update schedule: 247,150 chamber updates and 2,600 synthetic updates, interleaved deterministically.
- Update fractions: 98.959% chamber and 1.041% synthetic.
- Synthetic update: two synthetic image pairs accumulated before one optimizer step.
- Optimizer: AdamW, learning rate \(10^{-4}\), weight decay \(10^{-4}\), seed 734.
- Chamber loss: pixelwise L1.
- Synthetic loss: L1 + 0.02 mean-color loss + 0.008 residual-total-variation loss.
- Model output: the benchmark's direct RGB NAFNet construction with a
  sigmoid output wrapper. The residual-TV term regularizes the predicted
  change relative to the input; it is not a residual-output wrapper.
- Held-out chamber result: 24.408 dB PSNR, 0.79359 SSIM, and 0.04987 L1 over 552 pairs.
- Synthetic diagnostics: 19.395 dB / 0.78440 SSIM on 300 validation crops and 19.624 dB / 0.79092 SSIM on 500 test crops.

This run is primary because it preserves the same 2,600 synthetic-update exposure as the earlier sequential experiment while distributing those updates throughout chamber training.

## Sensitivity model

The completed 5x-synthetic run used 247,150 chamber updates and 13,000 synthetic updates:

- Held-out chamber: 24.440 dB / 0.79427 SSIM.
- Synthetic test diagnostic: 21.894 dB / 0.84067 SSIM.

It is treated only as a sensitivity check because its synthetic exposure is five times larger than the matched-budget design.

The 0.2x-synthetic run was incomplete when this paper package was frozen. It
later completed at 24.237 dB / 0.7928 SSIM on the chamber holdout and
16.315 dB / 0.7255 SSIM on the synthetic test diagnostic. The completed
config and summary are released with the GitHub experiment, but this audited
July 26 PDF was not retroactively expanded.

## Interpretation guardrails

- The synthetic data are included to suppress oversaturation and tiled-inference artifacts during transfer, not to maximize performance on the synthetic generator.
- Synthetic validation/test scores are generator-specific diagnostics, not a long-training benchmark and not evidence of real-fog generalization.
- The 30-model chamber benchmark remains the architecture-selection experiment.
- The mixed NAFNet is trained from scratch; it is not a fine-tuned chamber checkpoint.
- Existing task-specific O-HAZE/NH-HAZE and NTIRE models were initialized from the chamber checkpoint. They remain separate supervised adaptability checks and are not results of the mixed checkpoint.

## Validation status

- The adapted common evaluator passes all four unit tests.
- The evaluator preflight passes: all four NAFNet arms and all chamber,
  Mapillary, O-HAZE, NH-HAZE, and NTIRE paths are present.
- Both mixed checkpoints load strictly into the 29,159,715-parameter
  NAFNet. Embedded checkpoint metadata confirms 249,750 total optimizer
  steps for the primary model and 260,150 for the 5x sensitivity model,
  with no initialization checkpoint.
- A common public-dataset evaluation completed on every available
  paired image: O-HAZE (45), NH-HAZE (55), and the released NTIRE
  nighttime training pairs (25), for both completed mixed checkpoints.
  The primary model reached 17.061 / 0.6675 on O-HAZE, 12.903 / 0.4788
  on NH-HAZE, and 18.369 / 0.7335 on NTIRE (PSNR / SSIM). The 5x model
  reached 15.517 / 0.6497, 12.958 / 0.5071, and 18.533 / 0.7516,
  respectively. The paired 5x-minus-primary PSNR difference was
  -1.544 dB on O-HAZE (95% image-bootstrap CI -1.730 to -1.354);
  differences on NH-HAZE and NTIRE included zero. These are
  direct-transfer diagnostics, not supervised task-specific results,
  and are retained in `evaluation/REPORT_mixed_full_1024.md` rather
  than promoted to another manuscript benchmark table.
- The historical sequential arm is excluded from manuscript comparisons
  because it used a different output wrapper.
- A four-image, preselected O-HAZE/NH-HAZE visual comparison is available
  and is illustrative only; it is not a complete-dataset aggregate. The
  supplemental version compares hazy input, chamber-only NAFNet, the two
  completed mixed schedules, and ground truth. The figure was regenerated
  with embedded TrueType Times New Roman from
  `figure_sources/mixed_ratio_comparison/make_comparison.py`.
- Main Figure 2 was regenerated from `figure_sources/fig2_joint/make_fig2_joint_workflow.py` with genuine Times New Roman and installed in the draft main-paper package.
- The supplemental camera-sensitivity figure was regenerated from the
  existing 96-image chamber-checkpoint measurements with only the
  protocol-matched fog-chamber curve. This removes the historical
  sequential checkpoint, whose output wrapper differed. The copied data
  and generator are under `figure_sources/camera_sensitivity/`.
- The 43-image aircraft still set was reprocessed with the primary mixed
  checkpoint. Mean NIQE changed from 6.219 for the inputs to 6.407 for
  the outputs, with lower NIQE for 21 of 43 images. NIQE is therefore
  retained only as a descriptive statistic, not evidence of improved
  fog removal. The six-image main figure was rebuilt from these outputs.
- All 99 chamber-free fog-machine captures were reprocessed with the
  primary mixed checkpoint. Mean NIQE changed from 5.053 to 5.537, with
  lower NIQE for 34 of 99 outputs. The four-image supplemental figure is
  qualitative and explicitly notes color and tone changes.
- The mixed aircraft-video pipeline passed a two-frame smoke test and
  complete processing for both source MOV files, including HLG-to-sRGB
  conversion, checkpoint inference, side-by-side rendering, H.264/AAC
  encoding, and audio mapping. The validated outputs contain 1,007
  frames / 16.783 s and 446 frames / 7.433 s at 60 fps. Preview frames
  show stable spatial structure; the foggier clip also shows a
  persistent cyan/color shift, which is disclosed in the main text and
  supplemental video caption rather than hidden.
- Source QuickTime metadata identifies an iPhone 13 running software
  version 26.4.2. The draft's previous 26.5 value was corrected.
- The title was shortened from the causal “enables” construction to
  “for Cross-Domain Defogging.” Release language now distinguishes the
  already public fog-chamber dataset from the mixed checkpoints,
  scripts, and videos that will be added with the reproducibility
  package.
- The rewritten main and supplement compile successfully (13 and 16
  pages, respectively) with no unresolved references or overfull boxes.
  The final PDFs were rendered at 120 dpi and inspected page by page
  after the supplemental video captions and final wording corrections
  were added. No clipping, broken figures, or visible layout regressions
  were found.
- A clean handoff package is available under
  `deliverables/Mixed_Dataset_Joint_Training_Rewrite_20260726/`. Its
  stripped main and supplemental LaTeX source folders were independently
  rebuilt from copies in `/tmp`; both reproduced 13- and 16-page
  letter-size PDFs with no unresolved references, LaTeX warnings, or
  overfull boxes.

## Items deliberately left for author decision

- The disclosure statement still names Claude (Anthropic) only. It was
  not changed automatically because the appropriate wording for Codex
  assistance depends on the journal policy and the authors' preferred
  disclosure.
- The 0.2x-synthetic run completed after the manuscript package was frozen.
  Its result is documented in the GitHub experiment rather than inserted into
  this already-audited PDF.

## Backups

- `main/main_text.tex.20260726_1832`
- `supplement/defogging_supplement.tex.20260726_1832`
