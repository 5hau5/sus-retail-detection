
#call the Yolov10n model with weights
#Computes simple per-frame features from YOLO detections
# - Scores frames with either RF (predict_proba) or a linear rule (CMA-ES / DE / PSO)
    #return output text & confidence with video besides ./weights/outputs/    
    # output_text = model.predict(file_path)

from pathlib import Path
import os, json, math, time
import numpy as np
import cv2

# Optional dependencies (install if missing):
   # pip install ultralytics joblib opencv-python
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

try:
    import joblib
except Exception:
    joblib = None


# ---------------------------
# Configuration / file lookup
# ---------------------------

ROOT = Path(__file__).resolve().parent
WTS  = ROOT / "weights"
WTS.mkdir(exist_ok=True)

def _find_yolo_weights():
    # prefer files that look like your exported name, else fall back to best.pt
    cand = []
    cand += list(WTS.glob("yolo*best*.pt"))
    cand += list(WTS.glob("best.pt"))
    cand += list(WTS.glob("*.pt"))
    if not cand:
        raise FileNotFoundError("No YOLO .pt weights found in ./weights")
    return str(cand[0])

def _find_behavior_assets():
    """Returns a dict telling which behavior backend is available/selected."""
    # Priority: CMA-ES -> DE -> PSO -> RF (you can change preferred order)
    assets = {
        "rf_model": WTS / "rf_behavior.pkl",
        "scaler":   WTS / "behavior_scaler.pkl",
        "featjson": WTS / "behavior_features.json",
        "cma":      WTS / "cma_rule.json",
        "de":       WTS / "de_rule.json",
        "pso":      WTS / "pso_rule.json",
    }
    for k, p in assets.items():
        if p.exists() and p.is_file():
            assets[k] = p
        else:
            assets[k] = None

    # Choose backend:
    for backend in ("cma", "de", "pso", "rf"):  # change order if you like
        if backend == "rf":
            if assets["rf_model"] and assets["scaler"] and assets["featjson"]:
                return {"type": "rf", **assets}
        else:
            if assets[backend] and assets["scaler"] and assets["featjson"]:
                return {"type": backend, **assets}

    # Nothing found → use a simple heuristic later
    return {"type": "none", **assets}


# ------------------------------------
# Lazy singletons for speed and safety
# ------------------------------------

_YOLO = None
_BEHAVIOR = None  # dict w/ keys: type, objects, etc.

def _load_yolo():
    global _YOLO
    if _YOLO is not None:
        return _YOLO
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed. pip install ultralytics")
    ckpt = _find_yolo_weights()
    _YOLO = YOLO(ckpt)
    return _YOLO

def _sigmoid(x):  # for converting linear scores to a pseudo-probability
    return 1.0 / (1.0 + np.exp(-x))

def _load_behavior():
    """Load RF or linear-rule + scaler + feature list."""
    global _BEHAVIOR
    if _BEHAVIOR is not None:
        return _BEHAVIOR

    sel = _find_behavior_assets()
    btype = sel["type"]

    if btype == "rf":
        if joblib is None:
            raise RuntimeError("joblib is required for RF model. pip install joblib")
        rf = joblib.load(sel["rf_model"])
        scaler = joblib.load(sel["scaler"])
        feats = json.loads(Path(sel["featjson"]).read_text())["features"]
        _BEHAVIOR = {"type": "rf", "model": rf, "scaler": scaler, "features": feats}
        return _BEHAVIOR

    if btype in ("cma", "de", "pso"):
        rule = json.loads(Path(sel[btype]).read_text())  # {"w":[...],"tau":...}
        scaler = joblib.load(sel["scaler"]) if joblib else None
        feats = json.loads(Path(sel["featjson"]).read_text())["features"]
        _BEHAVIOR = {"type": "rule", "rule": rule, "scaler": scaler, "features": feats, "name": btype.upper()}
        return _BEHAVIOR

    # fallback: no behavior assets → we'll use a heuristic
    _BEHAVIOR = {"type": "none", "features": ["n_person","n_item","max_iou","min_center_dist"]}
    return _BEHAVIOR


# -------------------------------
# Feature computation (per frame)
# -------------------------------

def _xyxy_to_xywhn(box, W, H):
    x1,y1,x2,y2 = box
    w = max(0.0, float(x2 - x1))
    h = max(0.0, float(y2 - y1))
    cx = float(x1 + w/2.0) / max(1.0, W)
    cy = float(y1 + h/2.0) / max(1.0, H)
    return cx, cy, w / max(1.0, W), h / max(1.0, H)

def _iou_xywh(a, b):
    ax1, ay1 = a[0]-a[2]/2, a[1]-a[3]/2
    ax2, ay2 = a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1 = b[0]-b[2]/2, b[1]-b[3]/2
    bx2, by2 = b[0]+b[2]/2, b[1]+b[3]/2
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih  = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter   = iw*ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-9
    return inter / ua

def _center_dist(a, b):
    return math.dist((a[0],a[1]), (b[0],b[1]))

def _frame_features(result):
    """Compute the same interpretable features you used in the notebook."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {
            "n_person": 0, "n_item": 0,
            "areaP_mean": 0.0, "areaP_sum": 0.0,
            "areaI_mean": 0.0, "areaI_sum": 0.0,
            "min_center_dist": math.sqrt(2), "mean_center_dist": math.sqrt(2),
            "max_iou": 0.0, "sum_iou": 0.0
        }

    H, W = result.orig_img.shape[:2]
    xyxy = boxes.xyxy.cpu().numpy()
    cls  = boxes.cls.cpu().numpy().astype(int)

    pers_xywh = []
    item_xywh = []
    for b, c in zip(xyxy, cls):
        cx, cy, nw, nh = _xyxy_to_xywhn(b, W, H)
        if c == 0:    # person
            pers_xywh.append((cx,cy,nw,nh))
        elif c == 1:  # item
            item_xywh.append((cx,cy,nw,nh))

    areaP = [w*h for (_,_,w,h) in pers_xywh]
    areaI = [w*h for (_,_,w,h) in item_xywh]

    dists, ious = [], []
    for p in pers_xywh:
        for it in item_xywh:
            dists.append(_center_dist(p, it))
            ious.append(_iou_xywh(p, it))

    return {
        "n_person": len(pers_xywh),
        "n_item":   len(item_xywh),
        "areaP_mean": float(np.mean(areaP)) if areaP else 0.0,
        "areaP_sum":  float(np.sum(areaP))  if areaP else 0.0,
        "areaI_mean": float(np.mean(areaI)) if areaI else 0.0,
        "areaI_sum":  float(np.sum(areaI))  if areaI else 0.0,
        "min_center_dist": float(np.min(dists)) if dists else math.sqrt(2),
        "mean_center_dist": float(np.mean(dists)) if dists else math.sqrt(2),
        "max_iou": float(np.max(ious)) if ious else 0.0,
        "sum_iou": float(np.sum(ious)) if ious else 0.0
    }


# ----------------------------
# Behavior scoring per vector
# ----------------------------

def _score_feature_vector(feat_dict):
    B = _load_behavior()
    # Order features to match training
    names = B.get("features", list(feat_dict.keys()))
    x = np.array([feat_dict.get(k, 0.0) for k in names], dtype=np.float32).reshape(1,-1)

    if B["type"] == "rf":
        z = B["scaler"].transform(x)
        prob = float(B["model"].predict_proba(z)[0,1])
        return prob, "RF"

    if B["type"] == "rule":
        z = B["scaler"].transform(x) if B.get("scaler") is not None else x
        w = np.array(B["rule"]["w"], dtype=np.float32).reshape(1,-1)
        tau = float(B["rule"]["tau"])
        s = float(z @ w.T)          # linear score
        prob = float(_sigmoid(s - tau))  # pseudo-probability around the learned threshold
        return prob, B.get("name","RULE")

    # Fallback heuristic (if you shipped no behavior files):
    # - suspicious rises when a person is close to an item and IoU is non-zero
    hscore = 0.4 * (feat_dict["max_iou"] > 0.02) + \
             0.3 * (feat_dict["n_person"] >= 1 and feat_dict["n_item"] >= 1) + \
             0.3 * max(0.0, 1.0 - feat_dict["min_center_dist"])  # closer -> larger
    return float(hscore), "HEUR"


# --------------------------------
# Annotate frame and write a video
# --------------------------------

def _draw_overlay(frame, result, prob, backend_name):
    img = frame.copy()
    boxes = result.boxes
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        cls  = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        for (x1,y1,x2,y2), c, cf in zip(xyxy, cls, conf):
            color = (60,180,255) if c==0 else (80,255,120)  # person vs item
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            label = ("person" if c==0 else "item") + f" {cf:.2f}"
            cv2.putText(img, label, (x1, max(20,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    # risk banner
    txt = f"Risk ({backend_name}): {prob*100:.1f}%"
    color = (0,0,255) if prob >= 0.6 else (0,200,0)
    cv2.rectangle(img, (0,0), (img.shape[1], 28), (30,30,30), -1)
    cv2.putText(img, txt, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img

# --- simple tracking heuristics for concealment ---
NEAR_IOU = 0.18            # item overlaps person box a bit
NEAR_DIST = 0.10           # or centers close (on 0..1 normalized diag)
NEAR_MIN_FRAMES = 18        # need a small "near" streak first
CONCEAL_MISS_FRAMES = 12    # then item goes missing for N frames => conceal
CONCEAL_BOOST = 0.75       # floor risk when conceal is detected

def _iou_xyxy(a, b):
    # a,b: [x1,y1,x2,y2] in pixels
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
    inter = w * h
    area_a = max(0.0, (a[2]-a[0])) * max(0.0, (a[3]-a[1]))
    area_b = max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1]))
    union = area_a + area_b - inter + 1e-9
    return inter / union

def _center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def _norm_center_dist(pa, pb, W, H):
    # normalize by image diagonal so ~[0..1.4]
    dx = (pa[0]-pb[0]); dy = (pa[1]-pb[1])
    diag = (W**2 + H**2) ** 0.5
    return (dx*dx + dy*dy) ** 0.5 / (diag + 1e-9)

# -----------------------
# Public API for the app
# -----------------------

def run_inference(file_path: str,
                  conf_thresh: float = 0.25,
                  ema_alpha: float = 0.2,
                  stride: int = 1) -> dict:
    """
    Processes a video, writes an annotated MP4, and returns:
        {"text": "...", "confidence": float, "out_path": "path/to/video.mp4"}
    """
    model = _load_yolo()
    inp = Path(file_path)
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {file_path}")

    cap = cv2.VideoCapture(str(inp))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {file_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = WTS / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (inp.stem + "_annot.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    
    tracks = {}
    item_life = {}  # pid -> {'near_streak':int, 'miss':int, 'was_near':bool, 'concealed':bool}

    probs = []
    backend_name = "?"
    ema = 0.0
    fidx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if stride > 1 and (fidx % stride):
            fidx += 1
            continue

        
                # --- YOLO with tracking (keeps person IDs) ---
        results = model.track(frame, imgsz=640, conf=conf_thresh,
                              tracker="botsort.yaml", persist=True, verbose=False)[0]

        persons, items = [], []
        if results.boxes is not None and len(results.boxes) > 0:
            b = results.boxes
            xyxy = b.xyxy.cpu().numpy()
            cls  = b.cls.cpu().numpy().astype(int)
            conf = b.conf.cpu().numpy()
            ids  = b.id.cpu().numpy().astype(int) if b.id is not None else [None]*len(xyxy)

            # stricter for items
            CLASS_THRESH = {0: 0.25, 1: 0.55}     # person 0.25, item 0.55
            MIN_AREA = 0.0015                     # ignore tiny noise (<0.15% of frame)
            MAX_AREA = 0.35                       # ignore huge blobs (>35%)
            ITEM_PERSIST = 6                      # item must exist ≥6 consecutive frames

            seen_items_now = set()

            for i, bb in enumerate(xyxy):
                c = cls[i]; cf = conf[i]; tid = ids[i]
                area = ((bb[2]-bb[0])*(bb[3]-bb[1])) / float(W*H)

                # reject by conf/area
                if cf < CLASS_THRESH.get(c, 0.25) or area < MIN_AREA or area > MAX_AREA:
                    continue

                if c == 0:  # person
                    persons.append((tid, bb))
                elif c == 1:  # item
                    if tid is not None:
                        item_life[tid] = item_life.get(tid, 0) + 1
                        seen_items_now.add(tid)
                        if item_life[tid] >= ITEM_PERSIST:
                            items.append(bb)
                    else:
                        # if tracker didn't give ID, be extra strict
                        if cf >= 0.65:
                            items.append(bb)

                # decay counters for items not seen this frame
            for iid in list(item_life.keys()):
                if iid not in seen_items_now:
                    item_life[iid] = max(0, item_life[iid] - 1)
                    if item_life[iid] == 0:
                        item_life.pop(iid, None)

        # --- update per-person state, detect conceal ---
        conceal_triggered = False
        for pid, pbox in persons:
            if pid is None:
                continue
            st = tracks.setdefault(pid, {'near_streak':0, 'miss':0, 'was_near':False, 'concealed':False})

            # measure how close this person is to any item
            if items:
                pis = [_iou_xyxy(pbox, ib) for ib in items]
                pds = [_norm_center_dist(_center(pbox), _center(ib), W, H) for ib in items]
                max_iou = max(pis) if pis else 0.0
                min_dst = min(pds) if pds else 9.9
                near = (max_iou >= NEAR_IOU) or (min_dst <= NEAR_DIST)
            else:
                near = False

            if near:
                st['near_streak'] += 1
                st['miss'] = 0
            else:
                st['miss'] = st['miss'] + 1 if st['was_near'] else 0
                st['near_streak'] = max(0, st['near_streak'] - 1)

            st['was_near'] = near
            if (not st['concealed']) and st['near_streak'] >= NEAR_MIN_FRAMES and st['miss'] >= CONCEAL_MISS_FRAMES:
                st['concealed'] = True

            conceal_triggered = conceal_triggered or st['concealed']

        # --- base frame features + behavior score ---
        feats = _frame_features(results)
        p, backend_name = _score_feature_vector(feats)

        # boost if conceal detected on any track
        if conceal_triggered:
            p = max(p, CONCEAL_BOOST)

        # EMA smoothing + write frame
        ema = ema_alpha * p + (1.0 - ema_alpha) * ema
        probs.append(ema)

        vis = _draw_overlay(frame, results, ema, backend_name)
        writer.write(vis)

        fidx += 1

    cap.release()
    writer.release()

    if not probs:
        return {
            "text": "No frames processed.",
            "confidence": 0.0,
            "out_path": str(out_path)
        }

    # Final decision from top-k (more stable than single max)
    probs = np.array(probs, dtype=float)
    final_conf = float(np.percentile(probs, 90))  # robust high-risk percentile
    label = "Suspicious Activity: Possible shoplifting" if final_conf >= 0.6 else "Normal behavior"
    text = f"{label}\nConfidence: {final_conf*100:.1f}%"

    return {"text": text, "confidence": final_conf, "out_path": str(out_path)}


# ---------------------------------------------
# Backwards-compatible function for your UI.py
# ---------------------------------------------
def placeholder_function(file_path: str) -> str:
    """
    Kept for compatibility with your existing UI.
    Under the hood, runs the full pipeline and writes an annotated MP4.
    Returns text only (like your old stub). See run_inference() for richer output.
    """
    res = run_inference(file_path)
    return res["text"]
