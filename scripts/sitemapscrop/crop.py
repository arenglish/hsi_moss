from pathlib import Path
from PIL import Image


h = 3600
w = 2900

wr = 2
hr = 3
# impath = Path("scripts/sitemapscrop/M4242B.jp2")
# with Image.open(impath) as im:
#     t = 3903
#     l = 1700
#     r = 7262
#     im = im.crop((l, t, r, (r - l) * 3 / 2))
#     im = im.resize((w, h))
#     im.save("scripts/sitemapscrop/M4242B.crop.jpg")

impath = Path("scripts/sitemapscrop/M4242L.png")
with Image.open(impath) as im:
    t = 2100
    l = 700
    im = im.crop((l, t, l + w, t + h))
    im = im.resize((w, h))
    im.save("scripts/sitemapscrop/M4242L.crop.png")

impath = Path("scripts/sitemapscrop/finland.png")
with Image.open(impath) as im:
    t = 150
    l = 200
    r = 1100
    im = im.crop((l, t, r, (r - l) * 3 / 2))
    im = im.resize((w, h))
    im.save("scripts/sitemapscrop/finland.crop.png")
