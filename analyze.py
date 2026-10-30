# analyze.py : land use areas for the squares in water_tiles.gpkg
import csv, json, math
import numpy as np, rasterio, torch, geopandas as gpd
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from rasterio.windows import Window
from PIL import Image

# ---- settings: USE_NIR, DOWN and CROP must match what train.py used ----
SQUARES = "C:/Users/VasiP/Desktop/QGIS 3.44.14/LGN2025/Noorderplantsoen.gpkg"
IMG     = "C:/Users/VasiP/Desktop/QGIS 3.44.14/LGN2025/test_8cm.tif"
IMG_NIR = "C:/Users/VasiP/Desktop/QGIS 3.44.14/LGN2025/test_8cm_ir.tif"
USE_NIR = True
DOWN    = 2
CROP    = 512
# ------------------------------------------------------------------------

NAMES = {1: "building", 2: "road", 3: "grass", 4: "tree",
         5: "water", 6: "recreation", 7: "paved"}
PALETTE = {0: (0, 0, 0), 1: (255, 0, 0), 2: (255, 128, 0), 3: (128, 255, 0),
           4: (0, 128, 0), 5: (0, 0, 255), 6: (255, 0, 255), 7: (128, 128, 128)}
NC = len(NAMES)
CH = 4 if USE_NIR else 3
MODEL = "model_nir.pt" if USE_NIR else "model.pt"
dev = "cuda" if torch.cuda.is_available() else "cpu"

try:
    import postprocess as pp          # optional cleanup, used if the file is in the folder
except ImportError:
    pp = None

# normalization: same values as in training
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
if USE_NIR:                           # NIR statistics come from the training tiles
    tl, sp = json.load(open("tiles.json")), json.load(open("split.json"))
    vals = []
    with rasterio.open(IMG_NIR) as nir:
        for n in sp["train"]:
            r, c = tl[n]
            x = torch.from_numpy(nir.read(1, window=Window(c, r, 1875, 1875))).float() / 255
            if DOWN > 1:
                x = F.avg_pool2d(x[None, None], DOWN)[0, 0]
            vals.append(x.numpy().ravel()[::50])
    v = np.concatenate(vals)
    MEAN.append(float(v.mean())); STD.append(float(v.std()) + 1e-6)
MEAN = torch.tensor(MEAN).view(-1, 1, 1)
STD = torch.tensor(STD).view(-1, 1, 1)

model = smp.Unet("resnet34", encoder_weights=None, in_channels=CH, classes=NC + 1)
model.load_state_dict(torch.load(MODEL, map_location=dev))
model.to(dev).eval()
print("device:", dev, "| model:", MODEL, "| channels:", CH, "| cleanup:", pp is not None)


def predict(x):
    H, W = x.shape[1:]
    xt = torch.from_numpy(x).float() / 255
    if DOWN > 1:
        xt = F.avg_pool2d(xt[None], DOWN)[0]
    _, Hd, Wd = xt.shape
    stride = CROP // 2
    probs, cnt = torch.zeros(NC + 1, Hd, Wd), torch.zeros(Hd, Wd)
    rs = list(range(0, Hd - CROP, stride)) + [Hd - CROP]
    cs = list(range(0, Wd - CROP, stride)) + [Wd - CROP]
    with torch.no_grad():
        for r in rs:
            for c in cs:
                p = ((xt[:, r:r + CROP, c:c + CROP] - MEAN) / STD)[None].to(dev)
                pr = torch.softmax(model(p)[0], 0).cpu()
                pr[0] = 0
                probs[:, r:r + CROP, c:c + CROP] += pr
                cnt[r:r + CROP, c:c + CROP] += 1
    probs /= cnt
    if DOWN > 1:
        probs = F.interpolate(probs[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    return probs.argmax(0).numpy().astype(np.uint8)


sq = gpd.read_file(SQUARES)
rows, total = [], np.zeros(NC + 1)
tot_area = 0.0
with rasterio.open(IMG) as src:
    sq = sq.to_crs(src.crs)
    res, inv = src.res[0], ~src.transform
    for i, geom in enumerate(sq.geometry, start=1):
        minx, miny, maxx, maxy = geom.bounds
        c, r = inv * (minx, maxy)
        r, c = int(round(r)), int(round(c))
        h, w = round((maxy - miny) / res), round((maxx - minx) / res)
        if r < 0 or c < 0 or r + h > src.height or c + w > src.width:
            print(f"square {i}: outside the raster, skipped")
            continue
        win = Window(c, r, w, h)
        x = src.read([1, 2, 3], window=win)
        if USE_NIR:
            with rasterio.open(IMG_NIR) as nir:
                x = np.concatenate([x, nir.read(1, window=win)[None]], 0)
        pred = predict(x)
        if pp is not None:
            pred = pp.postprocess(pred)

        # EPSG:3857 stretches distances: true size = map size * cos(latitude)
        k = 1 / math.cosh(((miny + maxy) / 2) / 6378137.0)
        pix_m2 = (res * k) ** 2
        counts = np.bincount(pred.ravel(), minlength=NC + 1).astype(float)
        area = counts * pix_m2
        total += counts; tot_area += h * w * pix_m2
        rows.append([i, round(h * w * pix_m2)] + [round(a) for a in area[1:]])
        print(f"square {i}: " + ", ".join(f"{NAMES[k_]} {area[k_]:,.0f} m2" for k_ in NAMES))

        img = np.zeros((h, w, 3), np.uint8)
        for cid, col in PALETTE.items():
            img[pred == cid] = col
        Image.fromarray(img).save(f"landuse_square_{i}.png")

pix_total = total * (tot_area / total.sum())
print("\nTOTAL over all squares:")
for cid, cname in NAMES.items():
    print(f"  {cname:11s} {pix_total[cid]:>10,.0f} m2   {100 * total[cid] / total.sum():5.1f}%")
with open("land_use_areas.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["square", "square_m2"] + list(NAMES.values()))
    wr.writerows(rows)
    wr.writerow(["TOTAL", round(tot_area)] + [round(pix_total[c]) for c in NAMES])
print("saved land_use_areas.csv and landuse_square_<n>.png")