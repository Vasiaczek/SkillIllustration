# postprocess.py
"""
Cleans the raw predictions from train.py (pred_raw_<tile>.npy) using shape rules,
and scores them before and after. Needs no retraining, so settings can be tuned
in seconds. Needs: numpy, scipy, pillow.   pip install scipy

WHAT IT DOES
1. WATER AS ONE CONTINUOUS BODY
   A patch of non-water that is almost completely surrounded by water is
   relabeled as water, if it is either small, or thin (a strip). This removes
   "streets" painted in the middle of a canal. A bridge is NOT touched, because
   it touches land on both ends, so it is not surrounded by water.
2. BUILDINGS WITH CLEAN SHAPES
   a) Thin spurs and ragged edges (narrower than BUILD_OPEN px) are cut off.
   b) Optional (SNAP_BUILDINGS): near-rectangular buildings are replaced by
      their best-fit rectangle. Needs opencv:  pip install opencv-python-headless
3. SMALL SPECKLES
   Any patch smaller than its minimum area takes the label of its surroundings.
4. REPORT
   Prints the number of separate water and building pieces before and after.
   Fewer water pieces = more continuous waterways.
"""
import json
import numpy as np
from scipy import ndimage as ndi
from PIL import Image

# ---------------- settings (pixels at 8 cm; 1 m2 = about 156 px) ----------------
MIN_AREA = {1: 2000,    # building: ~13 m2
            2: 1500,    # road
            3: 1500,    # grass
            4: 1500,    # tree
            5: 3000,    # water: ~19 m2
            6: 1500,    # recreation
            7: 1500}    # paved
WATER_MAX_ISLAND = 20000    # non-water patch inside water, up to ~130 m2 -> water
WATER_THIN_HALF  = 40       # ...or a strip narrower than ~6 m -> water
WATER_ENCLOSED   = 0.8      # share of the patch border that must be water
BUILD_OPEN       = 9        # remove building parts thinner than ~70 cm
SNAP_BUILDINGS   = False    # replace near-rectangular buildings by a rectangle
SNAP_SOLIDITY    = 0.85
# ----------------------------------------------------------------------------------

BUILDING, WATER = 1, 5
NAMES = {1: "building", 2: "road", 3: "grass", 4: "tree",
         5: "water", 6: "recreation", 7: "paved"}
PALETTE = {0: (0, 0, 0), 1: (255, 0, 0), 2: (255, 128, 0), 3: (128, 255, 0),
           4: (0, 128, 0), 5: (0, 0, 255), 6: (255, 0, 255), 7: (128, 128, 128)}


def reassign(pred, remove, cls):
    """Give the pixels in `remove` the label of the nearest pixel that is not `cls`."""
    idx = ndi.distance_transform_edt(pred == cls, return_distances=False,
                                     return_indices=True)
    out = pred.copy()
    out[remove] = pred[idx[0][remove], idx[1][remove]]
    return out


def clean_small(pred, min_area):
    """Patches smaller than min_area take the most common label around them."""
    out = pred.copy()
    for c, m in min_area.items():
        lab, n = ndi.label(out == c)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        for i, sl in enumerate(ndi.find_objects(lab), start=1):
            if sizes[i] >= m:
                continue
            r0, r1 = max(sl[0].start - 3, 0), sl[0].stop + 3
            c0, c1 = max(sl[1].start - 3, 0), sl[1].stop + 3
            sub = lab[r0:r1, c0:c1] == i
            ring = ndi.binary_dilation(sub, iterations=2) & ~sub
            vals = out[r0:r1, c0:c1][ring]
            vals = vals[(vals != c) & (vals != 0)]
            if vals.size:
                out[r0:r1, c0:c1][sub] = np.bincount(vals).argmax()
    return out


def fill_water_gaps(pred):
    """Non-water patches that are enclosed by water, and small or thin, become water."""
    out = pred.copy()
    lab, n = ndi.label(pred != WATER)
    if n == 0:
        return out
    sizes = np.bincount(lab.ravel())
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        r0, r1 = max(sl[0].start - 3, 0), sl[0].stop + 3
        c0, c1 = max(sl[1].start - 3, 0), sl[1].stop + 3
        sub = lab[r0:r1, c0:c1] == i
        ring = ndi.binary_dilation(sub, iterations=2) & ~sub
        vals = pred[r0:r1, c0:c1][ring]
        if vals.size == 0 or (vals == WATER).mean() < WATER_ENCLOSED:
            continue
        thin = ndi.distance_transform_edt(np.pad(sub, 1)).max() <= WATER_THIN_HALF
        if sizes[i] <= WATER_MAX_ISLAND or thin:
            out[r0:r1, c0:c1][sub] = WATER
    return out


def clean_buildings(pred):
    """Cut thin spurs off buildings, optionally snap them to rectangles."""
    b = pred == BUILDING
    if BUILD_OPEN > 1 and b.any():
        opened = ndi.binary_opening(b, structure=np.ones((BUILD_OPEN, BUILD_OPEN), bool))
        pred = reassign(pred, b & ~opened, BUILDING)
    if SNAP_BUILDINGS:
        pred = snap_buildings(pred)
    return pred


def snap_buildings(pred):
    try:
        import cv2
    except ImportError:
        print("opencv not installed, skipping rectangle snapping")
        return pred
    out = pred.copy()
    lab, n = ndi.label(pred == BUILDING)
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        r0, c0 = max(sl[0].start - 2, 0), max(sl[1].start - 2, 0)
        r1, c1 = sl[0].stop + 2, sl[1].stop + 2
        sub = (lab[r0:r1, c0:c1] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        box = cv2.boxPoints(cv2.minAreaRect(cnt))
        rect_area = cv2.contourArea(box.astype(np.float32))
        if rect_area <= 0 or sub.sum() / rect_area < SNAP_SOLIDITY:
            continue                                   # not rectangular, leave it
        rect = np.zeros_like(sub)
        cv2.fillPoly(rect, [np.round(box).astype(np.int32)], 1)
        out[r0:r1, c0:c1][rect.astype(bool)] = BUILDING
    return out


def postprocess(pred):
    out = clean_small(pred, MIN_AREA)
    out = fill_water_gaps(out)
    out = clean_buildings(out)
    out = clean_small(out, MIN_AREA)
    return out


def n_pieces(pred, c):
    return ndi.label(pred == c)[1]


def score(pred, y, label):
    m = y > 0
    print(f"  {label}: pixel accuracy {(pred[m] == y[m]).mean():.3f}")
    for cid, cname in NAMES.items():
        t, p = (y == cid) & m, (pred == cid) & m
        if t.sum() == 0:
            continue
        print(f"    {cname:11s} IoU {(p & t).sum() / (p | t).sum():.2f}  "
              f"recall {(p & t).sum() / t.sum():.2f}")


if __name__ == "__main__":
    tiles = json.load(open("tiles.json"))
    split = json.load(open("split.json"))
    hand = np.load("hand.npy", mmap_mode="r")
    for name in split["test"]:
        raw = np.load(f"pred_raw_{name}.npy")
        r, c = tiles[name]
        y = np.array(hand[r:r + raw.shape[0], c:c + raw.shape[1]])
        clean = postprocess(raw)
        print(f"\ntile {name}")
        print(f"  water pieces:    {n_pieces(raw, WATER)} -> {n_pieces(clean, WATER)}")
        print(f"  building pieces: {n_pieces(raw, BUILDING)} -> {n_pieces(clean, BUILDING)}")
        score(raw, y, "before")
        score(clean, y, "after ")
        np.save(f"pred_clean_{name}.npy", clean)
        colored = np.zeros(clean.shape + (3,), np.uint8)
        for cid, col in PALETTE.items():
            colored[clean == cid] = col
        Image.fromarray(colored).save(f"pred_clean_{name}.png")
        print(f"  saved pred_clean_{name}.png")