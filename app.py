# app.py
import base64, io, math
from typing import List, Tuple, Dict
import numpy as np
import cv2
from flask import Flask, request, jsonify

app = Flask(__name__)


def _b64_to_cv(img_b64: str):
    raw = base64.b64decode(img_b64.split(",")[-1], validate=False)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from base64 string.")
    return img


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class DSU:
    def __init__(self, n):
        self.p = list(range(n)); self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]];
            x = self.p[x]
        return x

    def union(self, a, b):
        pa, pb = self.find(a), self.find(b)
        if pa == pb: return False
        if self.r[pa] < self.r[pb]: pa, pb = pb, pa
        self.p[pb] = pa
        if self.r[pa] == self.r[pb]: self.r[pa] += 1
        return True


def _mst_weight(n_nodes: int, edges: List[Tuple[int, int, int]]) -> int:
    if n_nodes <= 1:
        return 0  # MST weight for 0 or 1 node is 0

    edges = sorted(edges, key=lambda e: e[2])
    dsu = DSU(n_nodes)
    total = used = 0
    for u, v, w in edges:
        if dsu.union(u, v):
            total += int(w);
            used += 1
            if used == n_nodes - 1: break

    # Check if the graph is connected. If not, return 0 or handle as an error.
    # For this problem, let's assume if it's not connected, MST weight is 0
    # because a spanning tree cannot be formed.
    if used < n_nodes - 1 and n_nodes > 0:
        return 0  # Graph is not connected, cannot form a spanning tree

    return total


def _detect_nodes(img_bgr: np.ndarray) -> List[Tuple[int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)

    # Attempt 1: HoughCircles
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=25,
                               param1=100, param2=18, minRadius=6, maxRadius=28)
    out_hough = []
    if circles is not None:
        for c in np.uint16(np.around(circles))[0, :]:
            out_hough.append((int(c[0]), int(c[1]), int(c[2])))

    # Attempt 2: Contour-based detection for potentially missed nodes or if HoughCircles fails
    out_contours = []
    _, bw = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)
    bw = cv2.medianBlur(bw, 5)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        # Filter out very small or very large contours that are unlikely to be nodes
        if cv2.contourArea(c) < 30 or cv2.contourArea(c) > 2000:  # Adjust area thresholds as needed
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if 6 <= r <= 28:  # Use similar radius constraints as HoughCircles
            out_contours.append((int(x), int(y), int(r)))

    # Combine and deduplicate
    combined_nodes = out_hough + out_contours
    dedup = []
    for x, y, r in combined_nodes:
        # A more robust deduplication, checking against existing unique nodes
        is_duplicate = False
        for dx, dy, dr in dedup:
            if _dist((x, y), (dx, dy)) < 15:  # Tune this distance for merging close nodes
                is_duplicate = True
                break
        if not is_duplicate:
            dedup.append((x, y, r))

    return dedup


def _detect_line_segments(img_bgr: np.ndarray):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 50, 50)  # Stronger bilateral filter for noise reduction while preserving edges
    e = cv2.Canny(g, 50, 150, L2gradient=True)  # Slightly higher upper threshold for Canny
    segs = cv2.HoughLinesP(e, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)  # Adjusted thresholds
    return [] if segs is None else [tuple(map(int, s[0])) for s in segs]


def _segments_to_edges(nodes, segs, shape):
    centers = [(x, y) for x, y, _ in nodes]
    candidate_edges = []

    # Pre-calculate squared distances for efficiency
    node_sq_dists = {}
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            node_sq_dists[(i, j)] = _dist(centers[i], centers[j]) ** 2

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            a, b = centers[i], centers[j]
            # Skip if nodes are too close, they shouldn't be connected by a distinct edge
            if node_sq_dists[(i, j)] < 25 ** 2:  # 25 pixels, adjust as needed
                continue

            support = 0
            # Define a bounding box around the potential line segment for pre-filtering segments
            min_x, max_x = min(a[0], b[0]), max(a[0], b[0])
            min_y, max_y = min(a[1], b[1]), max(a[1], b[1])

            for x1, y1, x2, y2 in segs:
                # Basic check if segment is within the general vicinity
                seg_min_x, seg_max_x = min(x1, x2), max(x1, x2)
                seg_min_y, seg_max_y = min(y1, y2), max(y1, y2)

                # Check for overlap of bounding boxes (with a margin)
                if not (max_x + 10 > seg_min_x and min_x - 10 < seg_max_x and
                        max_y + 10 > seg_min_y and min_y - 10 < seg_max_y):
                    continue

                # Point-to-line distance check
                # Using OpenCV's pointPolygonTest for a more robust check if needed,
                # but the current `perp` logic is generally fine for straight lines.
                line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
                if line_len_sq == 0: continue  # Avoid division by zero

                # Project node A onto the line segment (x1,y1)-(x2,y2)
                t = ((a[0] - x1) * (x2 - x1) + (a[1] - y1) * (y2 - y1)) / float(line_len_sq)
                t = max(0, min(1, t))  # Clamp t to [0,1] to project onto segment

                closest_x = x1 + t * (x2 - x1)
                closest_y = y1 + t * (y2 - y1)

                # Distance from node A to the segment
                dist_a_to_seg = _dist(a, (closest_x, closest_y))

                # Project node B onto the line segment (x1,y1)-(x2,y2)
                t_b = ((b[0] - x1) * (x2 - x1) + (b[1] - y1) * (y2 - y1)) / float(line_len_sq)
                t_b = max(0, min(1, t_b))

                closest_x_b = x1 + t_b * (x2 - x1)
                closest_y_b = y1 + t_b * (y2 - y1)

                dist_b_to_seg = _dist(b, (closest_x_b, closest_y_b))

                # Check if both nodes are "close" to the segment
                # And if the segment roughly connects them
                if dist_a_to_seg < 10 and dist_b_to_seg < 10:  # Threshold for closeness
                    # Check if the segment is "between" the two nodes
                    # This is a heuristic and might need refinement
                    mid_seg_x, mid_seg_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    mid_nodes_x, mid_nodes_y = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0

                    if _dist((mid_seg_x, mid_seg_y), (mid_nodes_x, mid_nodes_y)) < _dist(a, b) / 2 + 10:
                        support += 1

            if support >= 1:  # If at least one supporting segment is found
                candidate_edges.append((i, j, ((centers[i][0] + centers[j][0]) // 2,
                                               (centers[i][1] + centers[j][1]) // 2)))
    return candidate_edges


_DIGIT_TEMPLATES = None


def _make_digit_templates():
    global _DIGIT_TEMPLATES
    if _DIGIT_TEMPLATES is not None: return _DIGIT_TEMPLATES
    # Added more scales and thicknesses for better robustness
    scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3];
    thicks = [1, 2, 3]
    templates = {d: [] for d in range(10)}
    for s in scales:
        for t in thicks:
            for d in range(10):
                canvas = np.zeros((40, 30), np.uint8)  # Increased canvas size for larger digits
                # Adjusted text position to be more centered
                cv2.putText(canvas, str(d), (int(2 + s * 2), int(32 + s * 2)), cv2.FONT_HERSHEY_SIMPLEX, s, 255, t,
                            cv2.LINE_AA)
                x, y, w, h = cv2.boundingRect((canvas > 0).astype(np.uint8))
                # Ensure a minimum size for the crop to avoid empty templates
                if w > 0 and h > 0:
                    crop = canvas[y:y + h, x:x + w]
                    if crop.size > 0: templates[d].append(crop)
    _DIGIT_TEMPLATES = templates
    return templates


def _match_digit(glyph: np.ndarray):
    tmpls = _make_digit_templates()
    if glyph is None or glyph.size == 0: return 0, -1.0

    # Pre-process glyph for better matching
    g = cv2.GaussianBlur(glyph, (3, 3), 0)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)

    best_d, best = -1, -1.0
    for d, vars in tmpls.items():
        for t in vars:
            th, tw = t.shape[:2]
            Gh, Gw = g.shape[:2]
            if min(Gh, Gw) <= 0: continue

            # Resize glyph to template size for comparison
            try:
                Gs = cv2.resize(g, (tw, th), interpolation=cv2.INTER_AREA)
            except cv2.error:  # Handle potential resizing errors
                continue

            res = cv2.matchTemplate(Gs, t, cv2.TM_CCOEFF_NORMED)
            score = float(res.max()) if res.size else -1.0
            if score > best: best, best_d = score, d
    return best_d, best


def _read_weight_near(img_bgr, pt):
    H, W = img_bgr.shape[:2];
    x, y = int(pt[0]), int(pt[1])

    # Dynamically adjust ROI size based on image size
    sz_factor = 0.08  # Increased factor for a larger search area
    sz = max(30, int(sz_factor * max(H, W)))

    x0, y0 = max(0, x - sz), max(0, y - sz);
    x1, y1 = min(W, x + sz), min(H, y + sz)
    roi = img_bgr[y0:y1, x0:x1]
    if roi.size == 0: return -1

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV);
    v = hsv[..., 2]
    # More robust thresholding for text: look for darker text on lighter background
    # or vice-versa. Here, assuming darker text on lighter background generally.
    _, mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Improve morphological operations
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=2)  # More iterations
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x2, y2, w2, h2 = cv2.boundingRect(c)
        # Refined aspect ratio and size filtering for digits
        aspect_ratio = w2 / float(h2)
        if 0.2 <= aspect_ratio <= 1.0 and 10 <= w2 <= 50 and 15 <= h2 <= 55:  # Adjusted dimensions
            boxes.append((x2, y2, w2, h2))

    if not boxes: return -1
    boxes.sort(key=lambda b: b[0])  # Sort by x-coordinate to read digits left-to-right

    digits = []
    for bx, by, bw, bh in boxes:
        digit_roi = cv2.cvtColor(roi[by:by + bh, bx:bx + bw], cv2.COLOR_BGR2GRAY)
        d, score = _match_digit(digit_roi)
        if score >= 0.45:  # Increased confidence threshold
            digits.append(str(d))

    if not digits: return -1
    try:
        val = int("".join(digits))
        # Add a plausible range check for weights if applicable to the problem
        if 0 < val < 1000:  # Example range, adjust as needed
            return val
        else:
            return -1  # Filter out implausible weights
    except ValueError:
        return -1


def _build_graph(img_bgr):
    nodes = _detect_nodes(img_bgr)
    if len(nodes) < 2: return len(nodes), []  # If less than 2 nodes, no edges possible

    segs = _detect_line_segments(img_bgr)
    candidate_edge_mids = _segments_to_edges(nodes, segs, img_bgr.shape)

    centers = [(x, y) for x, y, _ in nodes]
    edges_with_weights = []

    # Store found edges to avoid duplicate processing and for the "best weight" logic
    processed_edges = set()

    for i, j, mid_pt in candidate_edge_mids:
        # Sort node indices for consistent edge representation (u,v where u < v)
        u, v = (i, j) if i < j else (j, i)

        # Avoid reprocessing the same edge based on different segments
        if (u, v) in processed_edges:
            continue

        w = _read_weight_near(img_bgr, mid_pt)
        if w > 0:
            edges_with_weights.append((u, v, int(w)))
            processed_edges.add((u, v))

    # Fallback/refinement: If still not enough edges (e.g., lines were missed or
    # weights were hard to read), try to read weights between all pairs of nodes
    # that are relatively close, but only if they haven't been processed with a line segment.
    # This can introduce false positives if no line exists, so use with caution or refine.
    if len(edges_with_weights) < max(1, len(nodes) - 1):
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                u, v = (i, j) if i < j else (j, i)
                if (u, v) in processed_edges:
                    continue  # Skip if already processed

                # Heuristic: only try to read weight if nodes are "reasonably" close
                # This prevents creating edges between very distant nodes where no line likely exists
                if _dist(centers[u], centers[v]) < max(img_bgr.shape[0], img_bgr.shape[1]) / 3:  # Example threshold
                    mid = ((centers[i][0] + centers[j][0]) // 2, (centers[i][1] + centers[j][1]) // 2)
                    w = _read_weight_near(img_bgr, mid)
                    if w > 0:
                        edges_with_weights.append((u, v, int(w)))
                        processed_edges.add((u, v))

    # Dedup and find best weight for each unique edge
    best_weights_for_edges = {}
    for u, v, w in edges_with_weights:
        # Ensure consistent order (u,v where u<v)
        node_pair = (u, v) if u < v else (v, u)
        if node_pair not in best_weights_for_edges or w < best_weights_for_edges[node_pair]:
            best_weights_for_edges[node_pair] = w

    final_edges = [(u, v, w) for (u, v), w in best_weights_for_edges.items()]

    return len(nodes), final_edges


def _solve_single(img_b64: str) -> int:
    img = _b64_to_cv(img_b64)
    n, edges = _build_graph(img)
    return _mst_weight(n, edges)


@app.get("/healthz")
def healthz(): return "ok", 200


@app.post("/mst-calculation")
def mst_calculation():
    data = request.get_json(silent=True)
    if not isinstance(data, list) or not data:
        return jsonify({"error": "Body must be a JSON array of {image} objects"}), 400
    out = []
    for item in data:
        b64 = (item or {}).get("image")
        if not isinstance(b64, str):
            return jsonify({"error": "Each item must include an 'image' base64 string"}), 400
        try:
            val = _solve_single(b64)
        except Exception as e:
            # Log the exception for debugging in production
            print(f"Error processing image: {e}")
            val = 0  # Default to 0 on error
        out.append({"value": int(val)})
    return jsonify(out), 200


if __name__ == "__main__":
    import os

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))





