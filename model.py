# model.py  —  shared inference utilities for UI + offline detector
# Finds YOLO weights in ./weights, extracts per-frame features, scores risk,
# and draws overlays consistently.

from __future__ import annotations
from pathlib import Path
import os, json, math, time
import numpy as np
import cv2

# Optional deps
try:
    from ultralytics import YOLO as _YOLO_CLS
except Exception:
    _YOLO_CLS = None

try:
    import joblib
except Exception:
    joblib = None

# ---------------------------
# Paths / lazy singletons
# ---------------------------
ROOT = Path(__file__).resolve().parent
WTS  = ROOT / "weights"

_YOLO = None
_BEHAV = None  # dict holding backend & assets


# ---------------------------
# Weight discovery
# ---------------------------
def _find_yolo_weights() -> str:
    """Prefer yolo*best*.pt, then best.pt, then first *.pt under ./weights."""
    cand = []
    cand += list(WTS.glob("yolo*best*.pt"))
    cand += list(WTS.glob("best.pt"))
    cand += list(WTS.glob("*.pt"))
    if not cand:
        raise FileNotFoundError("No YOLO .pt weights found in ./weights")
    return str(cand[0])


def _find_behavior_assets():
    """Try RandomForest first, else linear rule JSON; else None."""
    rf_path   = WTS / "rf_behavior.pkl"
    scaler    = WTS / "behavior_scaler.pkl"
    feats_json= WTS / "behavior_features.json"
    # linear rules (any one is fine)
    rule_json = None
    for name in ("cma_rule.json", "de_rule.json", "pso_rule.json", "rule.json"):
        p = WTS / name
        if p.exists():
            rule_json = p
            break
    return {
        "rf": rf_path if rf_path.exists() else None,
        "scaler": scaler if scaler.exists() else None,
        "featnames": feats_json if feats_json.exists() else None,
        "rule": rule_json
    }


# ---------------------------
# Model loaders
# ---------------------------
def _load_yolo():
    """Load & cache YOLO model from ./weights."""
    global _YOLO
    if _YOLO is not None:
        return _YOLO
    if _YOLO_CLS is None:
        raise RuntimeError("ultralytics not installed")
    w = _find_yolo_weights()
    _YOLO = _YOLO_CLS(w)
    return _YOLO


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def _load_behavior():
    """Decide backend: RF (if files exist) else RULE (if json) else FALLBACK."""
    global _BEHAV
    if _BEHAV is not None:
        return _BEHAV

    found = _find_behavior_assets()
    backend = "fallback"
    obj = {"backend": backend}

    # 1) RandomForest head (preferred)
    if found["rf"] and joblib is not None:
        try:
            rf = joblib.load(found["rf"])
            scaler = joblib.load(found["scaler"]) if found["scaler"] else None
            featnames = json.loads(found["featnames"].read_text()) if found["featnames"] else None
            obj = {"backend": "rf", "rf": rf, "scaler": scaler, "featnames": featnames}
            _BEHAV = obj
            return obj
        except Exception:
            pass

    # 2) Linear rule from JSON
    if found["rule"]:
        try:
            rule = json.loads(found["rule"].read_text())
            # Expect {"w": {"feat": weight, ...}, "b": bias}
            if isinstance(rule, dict) and "w" in rule and "b" in rule:
                obj = {"backend": "rule", "rule": rule}
                _BEHAV = obj
                return obj
        except Exception:
            pass

    # 3) Fallback heuristic (no ext assets)
    _BEHAV = obj
    return obj


# ---------------------------
# Geometry helpers
# ---------------------------
def _xyxy_to_xywhn(x1,y1,x2,y2,W,H):
    w = max(1, x2-x1); h = max(1, y2-y1)
    cx = (x1 + x2)/2.0; cy = (y1 + y2)/2.0
    return (cx/W, cy/H, w/W, h/H)

def _iou_xywh(a, b):
    ax,ay,aw,ah = a; bx,by,bw,bh = b
    ax1, ay1 = ax-aw/2, ay-ah/2
    ax2, ay2 = ax+aw/2, ay+ah/2
    bx1, by1 = bx-bw/2, by-bh/2
    bx2, by2 = bx+bw/2, by+bh/2
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    ua = aw*ah + bw*bh - inter
    return inter/ua if ua>0 else 0.0

def _center_dist(a,b):
    ax,ay,aw,ah = a; bx,by,bw,bh = b
    return math.hypot(ax-bx, ay-by) / math.hypot(1.0,1.0)  # normalized diag


# ---------------------------
# Feature extraction per frame
# ---------------------------
def _frame_features(results) -> dict:
    """
    results: ultralytics Results (from model.track(...)[0])
    Returns a feature dict on 0..1-ish scales.
    """
    # Expect classes: 0=person, 1=item  (your data.yaml)
    boxes = getattr(results, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return {"n_person":0, "n_item":0, "max_iou":0.0, "min_dist":1.0,
                "mean_item_area":0.0, "pairs_near":0.0}

    xyxy = boxes.xyxy.cpu().numpy().astype(float) if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
    cls   = boxes.cls.cpu().numpy().astype(int)   if hasattr(boxes, "cls") and boxes.cls is not None else np.zeros(len(xyxy), int)
    W = int(results.orig_shape[1]); H = int(results.orig_shape[0])

    persons = [r for r,(c) in zip(xyxy, cls) if c==0]
    items   = [r for r,(c) in zip(xyxy, cls) if c==1]
    nP, nI = len(persons), len(items)

    max_iou = 0.0
    min_dist = 1.0
    pairs_near = 0.0
    areas = []

    if nI:
        for it in items:
            x1,y1,x2,y2 = it
            areas.append(((x2-x1)/W)*((y2-y1)/H))
    mean_item_area = float(np.mean(areas)) if areas else 0.0

    if nP and nI:
        for p in persons:
            px1,py1,px2,py2 = p
            Pn = _xyxy_to_xywhn(px1,py1,px2,py2,W,H)
            for it in items:
                ix1,iy1,ix2,iy2 = it
                In = _xyxy_to_xywhn(ix1,iy1,ix2,iy2,W,H)
                iou = _iou_xywh(Pn, In)
                d   = _center_dist(Pn, In)
                max_iou = max(max_iou, iou)
                min_dist = min(min_dist, d)
                if (iou >= 0.18) or (d <= 0.10):  # "near" heuristic
                    pairs_near += 1.0
        # normalize by possible pairs
        pairs_near = pairs_near / max(1.0, nP*nI)
    else:
        pairs_near = 0.0

    return {
        "n_person": float(nP),
        "n_item": float(nI),
        "max_iou": float(max_iou),
        "min_dist": float(min_dist),
        "mean_item_area": float(mean_item_area),
        "pairs_near": float(pairs_near),
    }


# ---------------------------
# Scoring backends
# ---------------------------
def _score_feature_vector(feats: dict) -> tuple[float,str]:
    """
    Returns (probability_on_[0..1], backend_name).
    """
    beh = _load_behavior()
    backend = beh["backend"]

    if backend == "rf":
        rf = beh["rf"]; scaler = beh["scaler"]; featnames = beh["featnames"]
        if featnames is None:
            # fallback to alphabetical order if names missing
            featnames = sorted(list(feats.keys()))
        x = np.array([[feats.get(k, 0.0) for k in featnames]], dtype=float)
        if scaler is not None:
            x = scaler.transform(x)
        try:
            proba = rf.predict_proba(x)[0,1]
        except Exception:
            # some classifiers use decision_function:
            s = float(rf.decision_function(x).ravel()[0])
            proba = _sigmoid(s)
        return float(proba), "rf"

    if backend == "rule":
        rule = beh["rule"]
        w = rule.get("w", {})
        b = float(rule.get("b", 0.0))
        s = b
        for k,v in w.items():
            s += float(v) * float(feats.get(k, 0.0))
        return _sigmoid(s), "rule"

    # fallback heuristic
    # Near pairs + item size around the person indicate risk
    s = 2.8*feats["pairs_near"] + 0.8*feats["max_iou"] + 0.4*(1.0 - feats["min_dist"]) + 0.3*feats["mean_item_area"]
    return _sigmoid(s - 1.2), "fallback"


# ---------------------------
# Overlay drawing
# ---------------------------
def _draw_overlay(img, results, prob: float, backend_name: str):
    vis = img.copy()
    boxes = getattr(results, "boxes", None)
    if boxes is not None and boxes.xyxy is not None:
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        cls  = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") and boxes.cls is not None else np.zeros(len(xyxy), int)
        ids  = boxes.id.cpu().numpy().astype(int) if hasattr(boxes, "id") and boxes.id is not None else [None]*len(xyxy)
        for (x1,y1,x2,y2),c,i in zip(xyxy, cls, ids):
            color = (0,0,255) if c==1 else (0,200,0)
            cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
            label = ("item" if c==1 else "person") + (f" #{i}" if i is not None else "")
            cv2.putText(vis, label, (int(x1)+3,int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # HUD banner
    txt = f"Risk ({backend_name}): {prob*100:.1f}%"
    color = (0,0,255) if prob >= 0.60 else (0,200,0)
    cv2.rectangle(vis, (0,0), (vis.shape[1], 28), (30,30,30), -1)
    cv2.putText(vis, txt, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return vis


# Convenience: export names for importers
__all__ = [
    "_load_yolo",
    "_frame_features",
    "_score_feature_vector",
    "_draw_overlay",
    "WTS",
]
