# train.py  (version 3: adds the infrared band)
"""
CHANGES FROM VERSION 1  (documentation)
=======================================
Problems seen in v1 on tile 48: streets painted inside waterways, many trees
missed, paved ~0 IoU, and no way to judge recreation (not in the test tile).

1. MORE CONTEXT (DOWN = 2)
   v1 looked at 384 px = ~30 m of ground at a time. A canal is defined by its
   shape (long, with a bank on both sides), which does not fit in that window.
   v2 shrinks the imagery 2x (8 cm -> 16 cm per pixel) and uses 512 px crops,
   so the model sees ~82 m. It is also ~4x cheaper per crop. Predictions are
   scaled back up to 8 cm, so outputs stay compatible with the labels.
   Set DOWN = 1 to go back to full resolution.

2. LONGER TRAINING WITH A LEARNING-RATE SCHEDULE
   400 steps -> 1500 steps (STEPS), with cosine decay of the learning rate so
   the loss settles instead of bouncing.

3. CLASS WEIGHTS FROM THE TRAINING TILES ONLY
   Rare classes (water, tree, paved, recreation) get a larger weight in the
   loss (square-root balanced, capped at 10). Computed from the training tiles
   only, so the test tile does not leak in.

4. EDGE-WEIGHTED LOSS (EDGE_WEIGHT = 2.0)
   Pixels within ~2 px of a boundary between two labeled classes count 3x in
   the loss. This pushes the model to draw sharp outlines, which matters for
   buildings. Set EDGE_WEIGHT = 0 to switch it off.

5. BRIGHTNESS AUGMENTATION
   Random +-15% brightness on each crop, to make shadows and dark water less
   confusing. Flips and 90-degree rotations are kept.

6. OVERLAPPING PREDICTION
   50% overlapping windows, probabilities averaged. Removes seams and gives
   every pixel context from all sides.

7. RAW PREDICTIONS ARE SAVED (pred_raw_<tile>.npy)
   postprocess.py reads them to fill water gaps and clean building shapes
   without retraining. Run it after this script.

8. TRAIN / PREDICT SWITCH
   TRAIN = False skips training and loads the saved model, so you can re-predict.


CHANGES FROM VERSION 2  (infrared band added)
=============================================
Why: in the version-2 results water (IoU 0.55) and paved (0.12) were still weak.
Water is almost black in near-infrared, vegetation is bright, and asphalt and
roofs fall in between, so this band separates exactly those classes.

9. FOURTH INPUT CHANNEL: NEAR-INFRARED (USE_NIR = True)
   - A second raster (IMG_NIR, the PDOK infrared layer, exported with the SAME
     extent, pixel size and CRS as the RGB raster) is read for every tile.
   - Band 1 of the infrared image is taken as NIR. PDOK's infrared product is a
     colour-infrared image, where near-infrared is shown in the red channel.
   - The model now has 4 input channels instead of 3. The pretrained ImageNet
     weights are adapted to 4 channels automatically by the library.
   - The NIR channel is normalized with the mean and standard deviation of the
     training tiles (RGB keeps the ImageNet values).
   - The script checks that both rasters have the same size and grid, and stops
     if they do not.
   - The model is saved as model_nir.pt, so the earlier RGB-only model.pt is
     not overwritten. Set USE_NIR = False to reproduce the version-2 behaviour.
   - Output files (pred_raw_*, pred_*, ...) use the same names as before and WILL
     overwrite the RGB-only results. Copy them to a folder first if you want to
     compare the two runs.
   - Nothing else changes: tiles, labels, split.json, postprocess.py and
     compare.py work as before. The painted tiles stay RGB.
"""
import json, random, time
import numpy as np, rasterio, torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from rasterio.windows import Window
from PIL import Image

# ---------------- settings ----------------
IMG         = "C:/Users/VasiP/Desktop/QGIS 3.44.14/LGN2025/test_8cm.tif"      # RGB raster, 15000 x 16875
IMG_NIR     = "C:/Users/VasiP/Desktop/QGIS 3.44.14/LGN2025/test_8cm_ir.tif"   # infrared raster, same grid
USE_NIR     = True     # True = 4 channels (RGB + near-infrared), False = RGB only
S           = 1875     # tile size in pixels at 8 cm
DOWN        = 2        # imagery shrink factor (2 = 16 cm per pixel, double the context)
CROP        = 512      # training crop, in shrunken pixels
STEPS       = 1500
BATCH       = 4
LR          = 3e-4
EDGE_WEIGHT = 2.0      # extra loss weight on class boundaries (0 = off)
TRAIN       = False     # False = load the saved model and only predict
SEED        = 0
# ------------------------------------------

NAMES = {1: "building", 2: "road", 3: "grass", 4: "tree",
         5: "water", 6: "recreation", 7: "paved"}
NC = len(NAMES)
CH = 4 if USE_NIR else 3                       # number of input channels
MODEL_PATH = "model_nir.pt" if USE_NIR else "model.pt"
PALETTE = {0: (0, 0, 0), 1: (255, 0, 0), 2: (255, 128, 0), 3: (128, 255, 0),
           4: (0, 128, 0), 5: (0, 0, 255), 6: (255, 0, 255), 7: (128, 128, 128)}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tiles = json.load(open("tiles.json"))
split = json.load(open("split.json"))
hand  = np.load("hand.npy", mmap_mode="r")
dev   = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

if USE_NIR:                                    # both rasters must share exactly the same grid
    with rasterio.open(IMG) as a, rasterio.open(IMG_NIR) as b:
        assert (a.width, a.height) == (b.width, b.height), \
            f"size differs: RGB {a.width}x{a.height}, infrared {b.width}x{b.height}"
        assert np.allclose(list(a.transform)[:6], list(b.transform)[:6], atol=1e-6), \
            "the infrared raster is not on the same grid (origin or pixel size differs)"
        assert a.crs == b.crs, "different CRS"
    print("RGB and infrared rasters share the same grid")


def load(name):
    """Return the shrunken image (CH,h,w) float 0..1, shrunken labels (h,w), full-res labels (S,S)."""
    r, c = tiles[name]
    win = Window(c, r, S, S)
    with rasterio.open(IMG) as src:
        x = src.read([1, 2, 3], window=win)                       # (3, S, S) uint8
    if USE_NIR:
        with rasterio.open(IMG_NIR) as nir:
            x = np.concatenate([x, nir.read(1, window=win)[None]], 0)   # (4, S, S): R, G, B, NIR
    y_full = np.array(hand[r:r + S, c:c + S])
    xt = torch.from_numpy(x).float() / 255
    if DOWN > 1:
        xd = F.avg_pool2d(xt[None], DOWN)[0]
        yd = y_full[::DOWN, ::DOWN][:xd.shape[1], :xd.shape[2]]
    else:
        xd, yd = xt, y_full
    return xd.numpy().astype(np.float32), np.ascontiguousarray(yd), y_full


train = []
for n in split["train"]:
    xd, yd, _ = load(n)
    train.append((xd, yd))
test = [(n,) + load(n) for n in split["test"]]
Sd = train[0][0].shape[1]
assert Sd >= CROP, f"CROP ({CROP}) is larger than the shrunken tile ({Sd})"
print("train tiles:", split["train"], "| test tiles:", split["test"], "| working size:", Sd,
      "| channels:", CH)

# ---- class weights from TRAIN tiles only ----
counts = np.zeros(NC + 1)
for _, yd in train:
    counts += np.bincount(yd.ravel(), minlength=NC + 1)
counts = counts[1:]
print("train pixels per class:", dict(zip(NAMES.values(), counts.astype(int))))
w = counts.sum() / np.maximum(counts, 1)
w = np.minimum(np.sqrt(w / w.min()), 10.0)
weights = torch.tensor([0.0] + list(w), dtype=torch.float32).to(dev)
print("class weights:", dict(zip(NAMES.values(), w.round(2))))

MEAN = torch.tensor([0.485, 0.456, 0.406])     # ImageNet values for R, G, B
STD  = torch.tensor([0.229, 0.224, 0.225])
if USE_NIR:                                    # NIR: statistics of the training tiles
    nir_px = np.concatenate([x[3].ravel()[::50] for x, _ in train])
    MEAN = torch.cat([MEAN, torch.tensor([float(nir_px.mean())])])
    STD  = torch.cat([STD,  torch.tensor([float(nir_px.std()) + 1e-6])])
    print(f"NIR channel: mean {float(nir_px.mean()):.3f}, std {float(nir_px.std()):.3f}")
MEAN, STD = MEAN.view(-1, 1, 1), STD.view(-1, 1, 1)


def prep(x):
    return (torch.from_numpy(np.ascontiguousarray(x)).float() - MEAN) / STD


def sample():
    for _ in range(20):                                   # skip crops that are mostly black
        x, y = random.choice(train)
        r = random.randint(0, Sd - CROP)
        c = random.randint(0, Sd - CROP)
        yc = y[r:r + CROP, c:c + CROP]
        if (yc > 0).mean() > 0.2:
            break
    xc = x[:, r:r + CROP, c:c + CROP]
    k = random.randint(0, 3)
    xc, yc = np.rot90(xc, k, (1, 2)), np.rot90(yc, k)
    if random.random() < 0.5:
        xc, yc = xc[:, :, ::-1], yc[:, ::-1]
    xc = np.clip(xc * random.uniform(0.85, 1.15), 0, 1)
    return prep(xc), torch.from_numpy(np.ascontiguousarray(yc)).long()


def loss_fn(logits, y):
    """Class-weighted cross entropy, with extra weight near class boundaries."""
    ce = F.cross_entropy(logits, y, weight=weights, ignore_index=0, reduction="none")
    if EDGE_WEIGHT > 0:
        yf = y.float().unsqueeze(1)
        mx = F.max_pool2d(yf, 5, 1, 2)
        mn = -F.max_pool2d(-yf, 5, 1, 2)
        edge = ((mx != mn) & (mn > 0)).float().squeeze(1)  # boundary between two real classes
        pw = 1.0 + EDGE_WEIGHT * edge
    else:
        pw = torch.ones_like(ce)
    return (ce * pw).sum() / (weights[y] * pw).sum().clamp_min(1.0)


try:
    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=CH, classes=NC + 1)
except Exception as e:
    print("Could not download ImageNet weights, training from scratch:", e)
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=CH, classes=NC + 1)
model = model.to(dev)

if TRAIN:
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    model.train()
    t0 = time.time()
    for step in range(1, STEPS + 1):
        xs, ys = zip(*[sample() for _ in range(BATCH)])
        x, y = torch.stack(xs).to(dev), torch.stack(ys).to(dev)
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 25 == 0:
            print(f"step {step}/{STEPS}  loss {loss.item():.3f}  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if step % 250 == 0:
            torch.save(model.state_dict(), MODEL_PATH)
    torch.save(model.state_dict(), MODEL_PATH)
else:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
    print("loaded", MODEL_PATH)


def predict(xd):
    """Overlapping windows, averaged probabilities, scaled back to full resolution."""
    _, Hd, Wd = xd.shape
    stride = CROP // 2
    probs = torch.zeros(NC + 1, Hd, Wd)
    cnt = torch.zeros(Hd, Wd)
    rs = list(range(0, Hd - CROP, stride)) + [Hd - CROP]
    cs = list(range(0, Wd - CROP, stride)) + [Wd - CROP]
    model.eval()
    with torch.no_grad():
        for r in rs:
            for c in cs:
                p = xd[:, r:r + CROP, c:c + CROP]
                pr = torch.softmax(model(prep(p)[None].to(dev))[0], 0).cpu()
                pr[0] = 0                                  # never predict "ignore"
                probs[:, r:r + CROP, c:c + CROP] += pr
                cnt[r:r + CROP, c:c + CROP] += 1
    probs /= cnt
    if DOWN > 1:
        probs = F.interpolate(probs[None], size=(S, S), mode="bilinear", align_corners=False)[0]
    return probs.argmax(0).numpy().astype(np.uint8)


# Test scoring: do this once at the end, and don't tune settings on these numbers.
for name, xd, _, y in test:
    pred = predict(xd)
    np.save(f"pred_raw_{name}.npy", pred)
    m = y > 0
    print(f"\ntile {name}: pixel accuracy {(pred[m] == y[m]).mean():.3f}")
    for cid, cname in NAMES.items():
        t, p = (y == cid) & m, (pred == cid) & m
        if t.sum() == 0:
            print(f"  {cname:11s} not in test tile")
            continue
        print(f"  {cname:11s} IoU {(p & t).sum() / (p | t).sum():.2f}  "
              f"recall {(p & t).sum() / t.sum():.2f}")
    colored = np.zeros((S, S, 3), np.uint8)
    for cid, col in PALETTE.items():
        colored[pred == cid] = col
    Image.fromarray(colored).save(f"pred_{name}.png")
    print(f"saved pred_{name}.png and pred_raw_{name}.npy")