# cut_tiles.py
import json
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.windows import Window
from PIL import Image

IMG   = "C:/Users/VasiP/Desktop/QGIS 3.44.14/LGN2025/test_8cm.tif"
TILES = "C:/Users/VasiP/Desktop/QGIS 3.44.14/LGN2025/TrainingData.gpkg"
S = 1875                                      # tile size in pixels

sel = gpd.read_file(TILES)

with rasterio.open(IMG) as src:
    print("raster:", src.width, "x", src.height, "| bands:", src.count)
    sel = sel.to_crs(src.crs)
    inv = ~src.transform
    tiles = {}
    for name, geom in zip(sel["id"], sel.geometry):
        minx, miny, maxx, maxy = geom.bounds
        c, r = inv * (minx, maxy)             # top-left corner in pixels
        r, c = int(round(r)), int(round(c))
        if r < 0 or c < 0 or r + S > src.height or c + S > src.width:
            raise ValueError(f"Tile {name} falls outside the raster")
        rgb = src.read([1, 2, 3], window=Window(c, r, S, S))   # first 3 bands only
        Image.fromarray(rgb.transpose(1, 2, 0)).save(f"tile_{name}.png")
        tiles[str(name)] = (r, c)
        print("saved", name, "row", r, "col", c)

    json.dump(tiles, open("tiles.json", "w"))
    json.dump({"transform": list(src.transform)[:6], "crs": src.crs.to_string(),
               "width": src.width, "height": src.height}, open("grid.json", "w"))
print("done:", len(tiles), "tiles")