import numpy as np, rasterio
from rasterio.warp import reproject, Resampling

LGN_PATH = "C:/Users/VasiP/Desktop/QGIS 3.44.14/LGN2025/GroningenData.tif"
IMG_PATH = "C:/Users/VasiP/Desktop/QGIS 3.44.14/DJ Crackhouse.tif"    # your aerial image of the AOI

LGN_TO_MY_CLASS = {
    16: "water", 17: "water",
    18: "built", 19: "built", 26: "built", 29: "built",
    251: "road", 252: "road", 253: "road",
    1: "grass", 23: "grass", 28: "grass", 45: "grass", 46: "grass", 47: "grass",
    2: "arable", 3: "arable", 4: "arable", 5: "arable", 6: "arable",
    8: "arable", 9: "arable", 10: "arable", 61: "arable", 62: "arable",
    11: "forest", 12: "forest", 20: "forest", 22: "forest", 40: "forest", 43: "forest",
    34: "heath", 36: "heath", 37: "heath", 38: "heath",
    321: "heath", 322: "heath", 323: "heath", 331: "heath", 332: "heath", 333: "heath",
    30: "sand", 31: "sand", 32: "sand", 33: "sand", 35: "sand",
    39: "wetland", 41: "wetland", 42: "wetland",
    27: "other",
}

classes = sorted(set(LGN_TO_MY_CLASS.values()))
class_id = {c: i + 1 for i, c in enumerate(classes)}          # 0 = ignore
lut = np.zeros(65536, dtype=np.uint8)
for code, name in LGN_TO_MY_CLASS.items():
    lut[code] = class_id[name]

with rasterio.open(IMG_PATH) as img, rasterio.open(LGN_PATH) as src:
    codes = np.zeros((img.height, img.width), dtype=src.dtypes[0])
    reproject(
        source=rasterio.band(src, 1), destination=codes,
        src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
        dst_transform=img.transform, dst_crs=img.crs, dst_nodata=0,
        resampling=Resampling.nearest,                         # never bilinear for classes
    )

mask = lut[codes]
np.save("mask.npy", mask)

print("class ids:", class_id)
print("unmapped codes in AOI:", sorted(set(np.unique(codes)) - set(LGN_TO_MY_CLASS) - {0}))
for name, i in class_id.items():
    print(f"{name:8s} {(mask == i).mean():.1%}")