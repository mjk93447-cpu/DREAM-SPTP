# FPCB Heatmap App GUI Update — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `fpcb_heatmap_app` GUI for the new version by adding an SOP-grade Help dialog, improving output usability, fixing any existing GUI bugs, and ensuring GitHub Actions produces a Windows EXE artifact.

**Architecture:** Keep existing processing pipeline (`FpcbProcessor`) and output formats. Add a `HelpDialog` (`QDialog`) with structured SOP content. Add output folder helpers (open folder, better logs) and tighten validation. Run functional verification on sample images and add a minimal automated “learning/test” sanity check of outputs.

**Tech Stack:** Python, PyQt5, OpenCV, (optional) Torch inference backend, GitHub Actions (Windows), PyInstaller.

---

## File Structure (target state)

**Modify:**
- `fpcb_heatmap_app/gui.py` — add Help button, Open Output button, wire dialog, improve validation/logging
- `fpcb_heatmap_app/README.md` — ensure it references updated workflow and run steps (if needed)
- `.github/workflows/fpcb-heatmap-build.yml` — adjust if new entrypoints/files are needed (keep existing artifact behavior)
- `.gitignore` — ensure local build/cache dirs aren’t committed (only if missing)

**Create:**
- `fpcb_heatmap_app/help_content.py` — SOP text (single source of truth, reused by dialog)
- `fpcb_heatmap_app/help_dialog.py` — `QDialog` implementation (tabs/sections)
- `fpcb_heatmap_app/tests/test_outputs_smoke.py` — smoke test validating output artifacts shape and JSON schema

## Task 1: Create Help content module (SOP text)

**Files:**
- Create: `fpcb_heatmap_app/help_content.py`

- [ ] **Step 1: Add SOP content as structured sections**

Create `fpcb_heatmap_app/help_content.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class HelpSection:
    title: str
    body_md: str


def get_help_sections() -> List[HelpSection]:
    return [
        HelpSection(
            title="1) What this tool does",
            body_md=(
                "This app generates crack/lead heatmaps and structured segment outputs from microscopic FPCB images.\n\n"
                "You provide an input image (or a folder of images) and an output folder. The app saves four outputs per run.\n"
            ),
        ),
        HelpSection(
            title="2) Before you start",
            body_md=(
                "Input requirements:\n"
                "- Supported formats: .png, .jpg/.jpeg, .bmp, .tif/.tiff\n"
                "- Recommendation: keep one product lot in one folder.\n\n"
                "Output folder:\n"
                "- Choose an empty folder for a clean run, or be aware that summary.csv is appended.\n"
            ),
        ),
        HelpSection(
            title="3) Step-by-step SOP",
            body_md=(
                "1. Click **Browse** next to **Input Image/Folder** and choose ONE image.\n"
                "   - If you cancel the file picker, you will be asked to pick a folder instead.\n"
                "2. Click **Browse** next to **Output Folder** and pick a folder.\n"
                "3. (Optional) Set **Model Checkpoint** if you were given a .pt/.pth file.\n"
                "4. Do NOT change numeric options unless instructed.\n"
                "5. Click **Start Processing** and wait until you see “Processing finished.”\n"
                "6. Click **Open Output Folder** and verify outputs.\n"
            ),
        ),
        HelpSection(
            title="4) Outputs checklist",
            body_md=(
                "For each input image `<name>.*`, the output folder must contain:\n"
                "- `<name>_overlay.png`\n"
                "- `<name>_heatmap.png`\n"
                "- `<name>_segments.json`\n\n"
                "Additionally, the folder run writes/updates:\n"
                "- `summary.csv` (NOTE: appended across runs)\n"
            ),
        ),
        HelpSection(
            title="5) Troubleshooting (FAQ)",
            body_md=(
                "**No images found**\n"
                "- Confirm you selected an image file or a folder containing supported formats.\n\n"
                "**Checkpoint not found**\n"
                "- Clear the checkpoint field or select a valid .pt/.pth path.\n\n"
                "**0 segments / empty results**\n"
                "- First rerun with defaults.\n"
                "- If still empty, try lowering Confidence Threshold slightly (e.g., 0.30 → 0.25).\n"
                "- If over-segmentation, increase Min Segment Length (e.g., 18 → 22).\n"
            ),
        ),
        HelpSection(
            title="6) Operational safety notes",
            body_md=(
                "- `summary.csv` is appended each run. For a clean report, move or delete it before running.\n"
                "- Avoid running with an output folder inside the input folder.\n"
            ),
        ),
    ]
```

## Task 2: Implement Help dialog (`QDialog`) and wire into GUI

**Files:**
- Create: `fpcb_heatmap_app/help_dialog.py`
- Modify: `fpcb_heatmap_app/gui.py`

- [ ] **Step 1: Create a tabbed Help dialog that renders SOP text**

Create `fpcb_heatmap_app/help_dialog.py`:

```python
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QListWidget, QListWidgetItem, QTextEdit, QVBoxLayout

from help_content import get_help_sections


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help — SOP (Field Engineer)")
        self.setMinimumSize(760, 520)

        root = QHBoxLayout()
        self.setLayout(root)

        self.nav = QListWidget()
        self.nav.setMaximumWidth(260)
        root.addWidget(self.nav)

        self.viewer = QTextEdit()
        self.viewer.setReadOnly(True)
        self.viewer.setLineWrapMode(QTextEdit.WidgetWidth)
        root.addWidget(self.viewer, 1)

        self.sections = get_help_sections()
        for s in self.sections:
            QListWidgetItem(s.title, self.nav)
        self.nav.currentRowChanged.connect(self._show_section)
        self.nav.setCurrentRow(0)

    def _show_section(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.sections):
            self.viewer.setPlainText("")
            return
        # Minimal “markdown-ish” rendering: use plain text with spacing preserved.
        # (PyQt5 QTextEdit supports rich text, but plain is most reliable for copy/paste.)
        self.viewer.setPlainText(self.sections[idx].body_md.replace("\\n", "\n"))
        self.viewer.moveCursor(self.viewer.textCursor().Start)
```

- [ ] **Step 2: Add Help button + Open Output Folder button in GUI**

Modify `fpcb_heatmap_app/gui.py`:

- Add imports:
```python
import subprocess
import sys
from PyQt5.QtCore import Qt
from help_dialog import HelpDialog
```

- Add buttons in `_init_ui` near Start/Stop:
```python
self.help_btn = QPushButton("Help")
self.help_btn.clicked.connect(self.open_help)
btn_layout.addWidget(self.help_btn)

self.open_out_btn = QPushButton("Open Output Folder")
self.open_out_btn.clicked.connect(self.open_output_folder)
btn_layout.addWidget(self.open_out_btn)
```

- Add methods:
```python
def open_help(self):
    dlg = HelpDialog(self)
    dlg.exec_()

def open_output_folder(self):
    out_dir = self.output_edit.text().strip()
    if not out_dir:
        QMessageBox.information(self, "Output Folder", "Please choose an output folder first.")
        return
    if not os.path.exists(out_dir):
        QMessageBox.warning(self, "Output Folder", "Output folder does not exist.")
        return
    try:
        if sys.platform.startswith("win"):
            os.startfile(out_dir)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", out_dir], check=False)
        else:
            subprocess.run(["xdg-open", out_dir], check=False)
    except Exception as exc:
        QMessageBox.warning(self, "Output Folder", f"Failed to open output folder: {exc}")
```

- [ ] **Step 3: Tighten validation messaging to match SOP**

In `start_processing`, adjust messages to be SOP-actionable:
- Missing input/output → “Click Browse …”
- Invalid numeric values → show which fields are invalid (see Task 3 for implementation detail)

## Task 3: Diagnose and fix GUI bugs / non-functional controls

**Files:**
- Modify: `fpcb_heatmap_app/gui.py`
- Possibly modify: `fpcb_heatmap_app/pipeline.py`, `fpcb_heatmap_app/io_utils.py` (only if a real bug is found)

- [ ] **Step 1: Run the GUI locally and click every control**

Run:
- `python fpcb_heatmap_app/main.py`

Checklist:
- Pick single image then run
- Pick folder then run
- Set invalid checkpoint path → correct error
- Enter invalid numeric values → correct error
- Stop while processing
- Help opens
- Open Output Folder opens

Expected:
- No exceptions printed to terminal
- Clear user messages

- [ ] **Step 2: Fix issues found**

Examples of likely issues to check/fix:
- Converting numeric fields might throw `ValueError` but does not tell user which field → implement per-field validation.
- Stop button only sets flag but GUI doesn’t reflect state → ensure finished state resets buttons even on early stop.
- Thread safety: log signals append rapidly → OK, but ensure thread is set to `None` on completion.

## Task 4: Add minimal automated output smoke test (“learning/training test”)

**Files:**
- Create: `fpcb_heatmap_app/tests/test_outputs_smoke.py`

- [ ] **Step 1: Create a test that runs `FpcbProcessor` with a synthetic image and validates outputs**

Create `fpcb_heatmap_app/tests/test_outputs_smoke.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from config import AppConfig
from pipeline import FpcbProcessor


def _write_synthetic_image(path: Path) -> None:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # draw a “lead-like” line and a “crack-like” line (just to exercise pipeline)
    cv2.line(img, (20, 120), (240, 120), (255, 255, 255), 2)
    cv2.line(img, (60, 60), (200, 200), (255, 255, 255), 1)
    ok = cv2.imwrite(str(path), img)
    assert ok


def test_outputs_smoke(tmp_path: Path):
    in_path = tmp_path / "sample.png"
    _write_synthetic_image(in_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = AppConfig(use_gpu=False, use_torch_backbone=False, output_dir=out_dir)
    proc = FpcbProcessor(cfg, checkpoint_path=None)
    result = proc.process_image(in_path, out_dir)

    assert result.overlay_path.exists()
    assert result.heatmap_path.exists()
    assert result.json_path.exists()
    assert (out_dir / "summary.csv").exists()

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert "meta" in payload
    assert "segments" in payload
    assert isinstance(payload["segments"], list)
```

- [ ] **Step 2: Add a test runner command**

Run from `fpcb_heatmap_app`:
- `python -m pip install -r requirements.txt`
- `python -m pip install pytest`
- `pytest -q`

Expected: PASS.

## Task 5: Ensure GitHub Actions workflow builds and uploads EXE artifact

**Files:**
- Modify: `.github/workflows/fpcb-heatmap-build.yml` (only if required)

- [ ] **Step 1: Confirm workflow builds from `fpcb_heatmap_app/main.py` and includes new modules**

Current build command uses:
- `pyinstaller --noconfirm --onefile --windowed --name fpcb-heatmap-app main.py`

Verify:
- New modules (`help_dialog.py`, `help_content.py`) are imported from `gui.py` and thus included automatically.

- [ ] **Step 2: Optionally bump Python version consistency**

If needed, keep `python-version: "3.11"` (recommended) and ensure dependencies install cleanly.

## Task 6: Git initialization, commits, and final verification

**Files:**
- Modify/create as needed: `.gitignore`, repo metadata

- [ ] **Step 1: Initialize git repo if missing**

Run:
- `git init`
- `git add .gitignore .github fpcb_heatmap_app docs`

- [ ] **Step 2: Commit spec + plan + implementation**

Use clear commit messages (multiple commits preferred):
- Commit 1: add spec + plan docs
- Commit 2: add Help dialog + GUI updates
- Commit 3: add tests + workflow adjustments

- [ ] **Step 3: Final local verification**

Run:
- `python -m pip install -r fpcb_heatmap_app/requirements.txt`
- `python fpcb_heatmap_app/main.py` (manual click-through)
- `python -m pip install pytest`
- `pytest -q`

Expected:
- GUI runs without exceptions
- Help opens and SOP matches behavior
- Smoke test passes

---

## Self-Review (plan vs spec)

- Spec coverage:
  - Help dialog (Task 1-2) ✅
  - Output usability (Task 2) ✅
  - Diagnose/fix GUI bugs (Task 3) ✅
  - Learning/training test (Task 4) ✅
  - Actions artifact (Task 5) ✅
  - GitHub commit/workflow (Task 6) ✅
- Placeholder scan: no TBD/TODO; commands and code included ✅
- Consistency: file paths and class names consistent (`HelpDialog`, `help_content`) ✅

