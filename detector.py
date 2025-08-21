# detector.py — offline analyzer that reuses model.py logic
from __future__ import annotations
import cv2, numpy as np
from collections import deque
from pathlib import Path
import model  # <— reuse the same functions as the UI

SMOOTH_WIN  = 20            # moving window for a softer final verdict
ALERT_RATIO = 0.35          # % of frames above thresh inside window to flag

def analyze_video(source, out_path=None, conf=0.25, imgsz=640):
    """
    Analyze a file/URL/camera and write an annotated MP4.
    Returns a summary dict with text, confidence, backend, and output path.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if out_path is None:
        out_dir = model.WTS / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (Path(str(source)).stem + "_annot.mp4")

    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H))

    yolo = model._load_yolo()
    win_flags = deque(maxlen=SMOOTH_WIN)
    prob_window = deque(maxlen=SMOOTH_WIN)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = yolo.track(
            frame, imgsz=imgsz, conf=conf,
            tracker="botsort.yaml", persist=True, verbose=False
        )[0]

        feats = model._frame_features(res)
        prob, backend = model._score_feature_vector(feats)
        prob_window.append(prob)
        win_flags.append(1 if prob >= 0.60 else 0)

        vis = model._draw_overlay(frame, res, prob, backend)
        vw.write(vis)

    cap.release()
    vw.release()

    if len(prob_window):
        avg_prob = float(np.mean(prob_window))
        alert_ratio = sum(win_flags) / max(1, len(win_flags))
    else:
        avg_prob = 0.0
        alert_ratio = 0.0

    label = "Suspicious Activity: Possible shoplifting" if (avg_prob >= 0.60 or alert_ratio >= ALERT_RATIO) else "Normal behavior"

    return {
        "text": f"{label}\nConfidence: {avg_prob*100:.1f}%",
        "confidence": avg_prob,
        "backend": backend if len(prob_window) else "n/a",
        "output_video": str(out_path)
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python detector.py <video_or_cam_index> [out.mp4]")
        sys.exit(1)
    src = sys.argv[1]
    try:
        src = int(src)
    except ValueError:
        pass
    out = sys.argv[2] if len(sys.argv) >= 3 else None
    summary = analyze_video(src, out)
    print(summary)
