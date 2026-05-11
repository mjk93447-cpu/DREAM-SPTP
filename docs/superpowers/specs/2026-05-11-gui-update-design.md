# GUI Update (FPCB Heatmap App) — Design

Last updated: 2026-05-11

## Purpose

Update the `fpcb_heatmap_app` PyQt5 GUI to match the “new version” scope:

- Add an in-app **Help** experience aimed at **field engineers**, written as a detailed SOP so a new operator can run it correctly by following steps.
- Improve GUI usability around outputs (open output folder, verify expected output files).
- While doing GUI work, **diagnose and fix** any existing GUI issues (buttons present but non-functional, incomplete features, or bugs found during testing).
- Verify final functionality and run “learning/training tests” (project-specific meaning: model/inference sanity checks + reproducible pipeline outputs).
- Commit to GitHub and ensure GitHub Actions builds and uploads an artifact.

## In Scope

### GUI changes

- Add a **Help** button to `fpcb_heatmap_app` GUI.
- Implement Help as a **dedicated `QDialog`** with:
  - A clear table-of-contents / sections navigation (tabs or list + stacked content).
  - Copy-friendly text (monospace blocks for file names / commands).
  - “What to check” output verification checklist.
- Add an **Open Output Folder** action (button) available after choosing an output directory (or always enabled when output path is valid).
- Improve input validation and user messaging:
  - Clear error prompts for missing/invalid paths.
  - Numeric option validation with actionable fixes.
  - Progress/log messaging that supports SOP execution.

### Help content (SOP)

Help content is for **field engineers** and must be written so “read and execute” is enough.

Required sections:

1. **What this tool does** (one paragraph, non-research phrasing)
2. **Before you start** (input image requirements; recommended folder structure)
3. **Step-by-step SOP** (numbered steps):
   - Select input image/folder
   - Select output folder
   - (Optional) set model checkpoint
   - Keep default parameters unless instructed
   - Start processing and wait for completion
4. **Outputs checklist** (must exist per image / per run):
   - `*_overlay.png`
   - `*_heatmap.png`
   - `*_segments.json`
   - `summary.csv` (batch summary; note that it appends across runs)
5. **Troubleshooting / FAQ**
   - No images found
   - Output files missing
   - Checkpoint not found / wrong file
   - Results empty (0 segments) and what knobs are safe to adjust
   - GPU/CPU fallbacks (if applicable)
6. **Operational safety notes**
   - `summary.csv` is appended (how to reset cleanly)
   - Where outputs are written and how to avoid overwriting

### Artifact / workflow

- Ensure a GitHub Actions workflow builds a Windows EXE artifact for `fpcb_heatmap_app` using PyInstaller.
- Artifact must be downloadable from Actions.

## Out of Scope (for this change)

- Major refactors of the inference/pipeline architecture not required for GUI+Help correctness.
- Updating `video_preprocessor` (explicitly considered complete).
- Changing output *formats* unless a bug requires it (keep same file extensions and payload shape; we can add backward-compatible fields if needed).

## Current Behavior (Baseline)

For each input image, `FpcbProcessor.process_image` produces:

- `<stem>_overlay.png`
- `<stem>_heatmap.png`
- `<stem>_segments.json` with keys:
  - `meta` (includes `image`, `engine`, `lead_score`, `crack_score`)
  - `segments` (list of `{label, points, length, score}`)
- `summary.csv` appended with:
  - `image`, `lead_segments`, `crack_segments`, `total_segments`, `lead_score`, `crack_score`

## Acceptance Criteria

- GUI contains a working **Help** button opening a structured SOP dialog.
- SOP is detailed, deterministic, and matches actual app behavior.
- “Open Output Folder” works on Windows and fails gracefully if path invalid.
- No GUI buttons are non-functional; any discovered GUI bugs are fixed.
- Functional verification is performed:
  - Run app on a small sample set (single image and folder) and confirm outputs.
  - Validate JSON schema and summary CSV generation.
- GitHub Actions produces an artifact for `fpcb_heatmap_app` build.

