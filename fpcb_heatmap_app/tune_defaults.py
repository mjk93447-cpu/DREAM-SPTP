from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path

import cv2

from config import AppConfig
from io_utils import list_images
from inference import SegmentationInference
from postprocess import collect_segments


def score_run(summary_rows: list[dict]) -> float:
    if not summary_rows:
        return -1e9
    # Heuristic:
    # - Prefer runs that detect both lead and crack
    # - Avoid over-segmentation by penalizing too many segments
    lead_nonzero = sum(1 for r in summary_rows if r["lead_segments"] > 0)
    crack_nonzero = sum(1 for r in summary_rows if r["crack_segments"] > 0)
    total_segments = sum(r["total_segments"] for r in summary_rows)
    mean_segments = total_segments / len(summary_rows)
    return (
        4.0 * lead_nonzero
        + 6.0 * crack_nonzero
        - 0.5 * abs(mean_segments - 12.0)
        - 0.03 * total_segments
    )


def main() -> None:
    sample_dir = Path("real_samples")
    images = list_images(str(sample_dir))
    base = AppConfig(
        confidence_threshold=0.38,
        min_segment_length=14.0,
        min_component_area=10,
        heatmap_blur_kernel=25,
        crack_weight=2.8,
        lead_weight=1.0,
        use_gpu=False,
        use_torch_backbone=False,
    )

    grid = {
        "confidence_threshold": [0.30, 0.35, 0.40],
        "min_segment_length": [10.0, 14.0, 18.0],
        "min_component_area": [8, 12, 16],
        "heatmap_blur_kernel": [21, 25],
        "crack_weight": [2.4, 2.8, 3.2],
    }

    best = None
    idx = 0
    for conf, min_len, min_area, blur, crack_w in product(
        grid["confidence_threshold"],
        grid["min_segment_length"],
        grid["min_component_area"],
        grid["heatmap_blur_kernel"],
        grid["crack_weight"],
    ):
        idx += 1
        cfg = replace(
            base,
            confidence_threshold=conf,
            min_segment_length=min_len,
            min_component_area=min_area,
            heatmap_blur_kernel=blur,
            crack_weight=crack_w,
        )
        infer = SegmentationInference(
            checkpoint_path=None,
            confidence_threshold=cfg.confidence_threshold,
            use_gpu=cfg.use_gpu,
            use_torch_backbone=cfg.use_torch_backbone,
        )
        rows: list[dict] = []
        for image in images:
            image_bgr = cv2.imread(str(image))
            if image_bgr is None:
                continue
            out = infer.predict(image_bgr)
            segments = collect_segments(
                lead_mask=out.lead_mask,
                crack_mask=out.crack_mask,
                min_area=cfg.min_component_area,
                min_length=cfg.min_segment_length,
                lead_score=out.lead_score,
                crack_score=out.crack_score,
            )
            lead_count = sum(1 for s in segments if s.label == "lead")
            crack_count = sum(1 for s in segments if s.label == "crack")
            rows.append(
                {
                    "lead_segments": lead_count,
                    "crack_segments": crack_count,
                    "total_segments": len(segments),
                }
            )
        run_score = score_run(rows)
        candidate = (run_score, cfg, rows)
        if best is None or run_score > best[0]:
            best = candidate
        print(
            f"run={idx:03d} score={run_score:.3f} "
            f"conf={conf} min_len={min_len} min_area={min_area} blur={blur} crack_w={crack_w}"
            ,
            flush=True,
        )

    assert best is not None
    print("\nBEST CONFIG")
    print(best[1])
    print("ROWS", best[2])


if __name__ == "__main__":
    main()

