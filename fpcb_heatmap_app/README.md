# FPCB Microscopic Heatmap App

PyQt5 desktop app for generating 2D heatmaps from microscopic FPCB images.

## Features

- Single image or folder batch processing
- DL-assisted lead/crack mask inference with OpenCV fallback
- Line-segment extraction from masks via skeletonization
- Outputs:
  - `<name>_overlay.png`
  - `<name>_heatmap.png`
  - `<name>_segments.json`
  - `summary.csv`

## Tuned Default Preset (Ready To Use)

The current defaults were tuned with real-world microscopic copper trace crack samples from:
- [Micromachines 2025 (PMC Open Access)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12029595/)

Applied startup defaults:
- `confidence_threshold=0.30`
- `min_segment_length=18`
- `min_component_area=8`
- `heatmap_blur_kernel=21`
- `crack_weight=2.4`
- `lead_weight=1.0`
- `use_torch_backbone=False` (fast startup, no pretrained weight download)

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Build (Windows)

```bash
pip install pyinstaller
pyinstaller --onefile --windowed main.py --name fpcb-heatmap-app
```

## GitHub Actions EXE Artifact

Workflow file:
- `.github/workflows/fpcb-heatmap-build.yml`

After push to `main` (or manual `workflow_dispatch`), download:
- Actions -> `Build FPCB Heatmap EXE` -> latest run -> artifact `fpcb-heatmap-app-exe`

