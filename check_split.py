import json
S = 1875
tiles = json.load(open("tiles.json"))
split = json.load(open("split.json"))
for t in split["test"]:
    r, c = tiles[t]
    touching = [n for n in split["train"]
                if abs(r - tiles[n][0]) <= S and abs(c - tiles[n][1]) <= S]
    print(t, "touches:", touching if touching else "nothing (good)")