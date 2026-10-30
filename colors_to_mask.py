# colors_to_mask.py
import os, json
import numpy as np
from PIL import Image

S = 1875

PALETTE = {1: (255, 0, 0),     # building
           2: (255, 128, 0),   # road
           3: (128, 255, 0),   # grass
           4: (0, 128, 0),     # tree
           5: (0, 0, 255),     # water
           6: (255, 0, 255),   # recreation
           7: (128, 128, 128)} # paved
NAMES = {1: "building", 2: "road", 3: "grass", 4: "tree",
         5: "water", 6: "recreation", 7: "paved"}

tiles = json.load(open("tiles.json"))                  # name: [row, col]
g = json.load(open("grid.json"))
H, W = g["height"], g["width"]
hand = np.zeros((H, W), np.uint8)                      # 0 = unlabeled / ignored

done = []
for name, (r, c) in tiles.items():
    path = f"label_{name}.png"
    if not os.path.exists(path):
        continue                                       # tile not painted yet
    lab = np.array(Image.open(path).convert("RGB"), np.int32)
    assert lab.shape[:2] == (S, S), f"{name}: wrong size {lab.shape}"

    blank = lab.sum(-1) == 0                           # pure black = unlabeled
    best_d = np.full((S, S), np.inf)
    mask = np.zeros((S, S), np.uint8)
    for cid, col in PALETTE.items():                   # nearest palette colour, one class at a time
        d = ((lab - np.array(col)) ** 2).sum(-1)
        better = d < best_d
        mask[better] = cid
        best_d[better] = d[better]
    mask[blank] = 0

    snapped = ((best_d > 0) & ~blank).mean()
    print(f"{name}: {(~blank).mean():.0%} labeled, {snapped:.2%} snapped")
    hand[r:r+S, c:c+S] = mask
    done.append(name)

np.save("hand.npy", hand)
print("converted:", done)
total = (hand > 0).sum()
for cid, cname in NAMES.items():
    n = (hand == cid).sum()