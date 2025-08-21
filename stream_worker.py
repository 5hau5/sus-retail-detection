# stream_worker.py
from __future__ import annotations
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
import cv2, numpy as np
import model

class InferenceThread(QThread):
    frame_ready = pyqtSignal(object)   # QImage
    hud_text    = pyqtSignal(str)
    finished    = pyqtSignal(dict)

    def __init__(self, source_path, stride=1, ema_alpha=0.20, conf=0.25, parent=None):
        super().__init__(parent)
        self.source_path = source_path
        self.stride = stride
        self.ema_alpha = ema_alpha
        self.conf = conf
        self._stop = False

    def stop(self): self._stop = True

    def run(self):
        from PyQt6.QtGui import QImage
        m = model._load_yolo()
        cap = cv2.VideoCapture(self.source_path)
        if not cap.isOpened():
            self.finished.emit({"error": f"Cannot open: {self.source_path}"})
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        out_dir = (model.WTS / "outputs"); out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (f"{Path(self.source_path).stem}_annot.mp4")
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

        ema, probs, fidx = 0.0, [], 0
        backend_name = "?"
        while not self._stop:
            ok, frame = cap.read()
            if not ok: break
            if self.stride > 1 and (fidx % self.stride): fidx += 1; continue

            results = m.track(frame, imgsz=640, conf=self.conf, tracker="botsort.yaml", persist=True, verbose=False)[0]
            feats = model._frame_features(results)
            p, backend_name = model._score_feature_vector(feats)
            ema = self.ema_alpha * p + (1.0 - self.ema_alpha) * ema
            probs.append(ema)
            vis = model._draw_overlay(frame, results, ema, backend_name)
            writer.write(vis)

            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            self.frame_ready.emit(qimg)
            self.hud_text.emit(f"Risk ({backend_name}): {ema*100:.1f}%")
            fidx += 1

        cap.release(); writer.release()
        final_conf = float(np.percentile(np.array(probs, dtype=float), 90)) if probs else 0.0
        label = "Suspicious Activity: Possible shoplifting" if final_conf >= 0.6 else "Normal behavior"
        self.finished.emit({"text": f"{label}\nConfidence: {final_conf*100:.1f}%",
                            "confidence": final_conf, "output_video": str(out_path), "backend": backend_name})
