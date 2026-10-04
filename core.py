"""Photo-processing core for Deluxe Digital Studio (Android port)."""
import os, threading
import numpy as np, cv2
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageEnhance

HERE = os.path.dirname(os.path.abspath(__file__))
DPI = 600
PREVIEW_SCALE = 0.3
def mm(v): return round(v / 25.4 * DPI)
PH_W, PH_H = mm(35), mm(45)
SH_W, SH_H = mm(152.4), mm(101.6)

BG_COLOR = (70, 170, 220)
CHANGE_BG = True
EXPOSURE = 1.03
CONTRAST = 1.0
COLOR = 1.0
SHARPEN = True
CROP_H_FACTOR = 3.25
CROP_FACE_Y = 0.42
BORDER = max(1, mm(0.4))

def _casc(name):
    return cv2.CascadeClassifier(os.path.join(HERE, name))
FACE = _casc("haarcascade_frontalface_default.xml")
FACE2 = _casc("haarcascade_frontalface_alt2.xml")
EYE = _casc("haarcascade_eye.xml")

def open_photo(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def enhance(img, scale=1.0):
    small = np.asarray(img.resize((256, max(1, round(256 * img.height / img.width))), Image.BOX))
    m = small.reshape(-1, 3).mean(0)
    gains = np.clip(m.mean() / np.maximum(m, 1e-3), 0.96, 1.04)
    if np.abs(gains - 1).max() > 1e-3:
        img = Image.merge("RGB", [band.point([min(255, int(v * g + 0.5)) for v in range(256)])
                                  for band, g in zip(img.split(), gains)])
    if EXPOSURE != 1.0:
        img = ImageEnhance.Brightness(img).enhance(EXPOSURE)
    if CONTRAST != 1.0:
        img = ImageEnhance.Contrast(img).enhance(CONTRAST)
    if COLOR != 1.0:
        img = ImageEnhance.Color(img).enhance(COLOR)
    if SHARPEN:
        img = img.filter(ImageFilter.UnsharpMask(max(0.3, scale), 40, 3))
    return img


def blue_background(img, face):
    sw = 512
    sh = round(sw * img.height / img.width)
    k = sw / img.width
    small = cv2.cvtColor(np.asarray(img.resize((sw, sh), Image.LANCZOS)), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    cx, cy, fh = face[0] * k, face[1] * k, face[2] * k
    yy, xx = np.mgrid[0:sh, 0:sw]

    head = ((xx - cx) / (0.9 * fh)) ** 2 + ((yy - (cy - 0.2 * fh)) / (1.2 * fh)) ** 2 < 1
    zone = head.astype(np.uint8)
    body = np.array([[cx - 1.7 * fh, sh], [cx + 1.7 * fh, sh], [cx + 0.95 * fh, cy + 0.55 * fh],
                     [cx - 0.95 * fh, cy + 0.55 * fh]], np.int32)
    cv2.fillPoly(zone, [body], 1)
    fgseed = np.zeros((sh, sw), np.uint8)
    cv2.ellipse(fgseed, (int(cx), int(cy + 0.05 * fh)), (int(0.32 * fh), int(0.5 * fh)), 0, 0, 360, 1, -1)
    cv2.rectangle(fgseed, (int(cx - 0.4 * fh), int(cy + 0.75 * fh)), (int(cx + 0.4 * fh), sh), 1, -1)

    samp = np.zeros((sh, sw), bool)
    mx, my = int(0.10 * sw), int(0.06 * sh)
    samp[:int(0.6 * sh), :mx] = True
    samp[:int(0.6 * sh), sw - mx:] = True
    samp[:my, :] = True
    samp &= ~head
    pts = lab[samp]
    if len(pts) < 300:
        return img, False
    K = 4
    _, lbl, cen = cv2.kmeans(pts, K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                             3, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.ravel()
    wts = np.array([0.5, 1.0, 1.0], np.float32)
    dmin = np.full((sh, sw), 1e9, np.float32)
    for i in range(K):
        mem = pts[lbl == i]
        if len(mem) < 0.08 * len(pts):
            continue
        di = np.sqrt((((mem - cen[i]) * wts) ** 2).sum(1))
        thr = max(9.0, float(di.mean() + 3 * di.std()))
        dmin = np.minimum(dmin, np.sqrt((((lab - cen[i]) * wts) ** 2).sum(2)) / thr)
    cand = (dmin < 1.0).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    outside = cand.copy()
    outside[zone > 0] = 0
    n, cc = cv2.connectedComponents(outside)
    edge = set(np.unique(np.concatenate([cc[0, :], cc[:, 0], cc[:, -1], cc[-1, :]]))) - {0}
    sure_bg = np.isin(cc, list(edge)).astype(np.uint8)
    if sure_bg.mean() < 0.08:
        return img, False
    sure_bg_in = cv2.erode(sure_bg, np.ones((5, 5), np.uint8))

    m = np.full((sh, sw), cv2.GC_PR_FGD, np.uint8)
    m[cand > 0] = cv2.GC_PR_BGD
    m[sure_bg_in > 0] = cv2.GC_BGD
    m[fgseed > 0] = cv2.GC_FGD
    dark = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 2] < 70
    m[head & dark & (cand == 0)] = cv2.GC_FGD
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(small, m, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return img, False
    fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)

    n, lab_cc = cv2.connectedComponents(fg)
    lbl0 = lab_cc[min(max(int(cy), 0), sh - 1), min(max(int(cx), 0), sw - 1)]
    if lbl0 == 0:
        return img, False
    fg = (lab_cc == lbl0).astype(np.uint8)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg = np.zeros_like(fg)
    cv2.drawContours(fg, cnts, -1, 1, -1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, ker)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, ker)

    c = max(4, int(0.06 * sw))
    corners = np.concatenate([fg[:c, :c].ravel(), fg[:c, -c:].ravel()])
    leak = (fg & sure_bg_in).sum() / max(1, sure_bg_in.sum())
    if not (0.12 < fg.mean() < 0.9) or corners.mean() > 0.2 or leak > 0.1:
        return img, False

    fg = cv2.erode(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    alpha = cv2.GaussianBlur(fg.astype("float32"), (0, 0), 1.4)
    alpha = np.clip((alpha - 0.5) * 1.8 + 0.5, 0, 1)
    a = np.asarray(img).astype("float32")
    q = 4
    size = (max(8, img.width // q), max(8, img.height // q))
    core = (alpha > 0.99).astype("uint8")
    core = cv2.erode(core, np.ones((3, 3), np.uint8)).astype("float32")
    core_s = (cv2.resize(core, size, interpolation=cv2.INTER_AREA) > 0.99).astype("float32")
    sm = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    sig = max(0.6, 3.0 * img.width / PH_W)
    num = cv2.GaussianBlur(sm * core_s[..., None], (0, 0), sig)
    den = cv2.GaussianBlur(core_s, (0, 0), sig)[..., None]
    fgc = cv2.resize(num / np.maximum(den, 1e-3), (img.width, img.height), interpolation=cv2.INTER_CUBIC)
    alpha_f = cv2.resize(alpha, img.size, interpolation=cv2.INTER_CUBIC)[..., None].clip(0, 1)
    t = np.clip((alpha_f - 0.7) / 0.3, 0, 1)
    solid = a * t + fgc * (1 - t)
    out = solid * alpha_f + np.array(BG_COLOR, "float32") * (1 - alpha_f)
    return Image.fromarray(out.clip(0, 255).astype("uint8")), True


MIN_CROP_W = round(35 / 25.4 * 300)


def _find_face(gray, rgb=None):
    H, W = gray.shape
    g = cv2.equalizeHist(gray)
    ms = max(40, int(0.09 * H))
    skin = None
    if rgb is not None:
        ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        skin = (ycc[..., 1] >= 133) & (ycc[..., 1] <= 173) & (ycc[..., 2] >= 77) & (ycc[..., 2] <= 127)
    boxes = []
    for casc, nb, wgt in ((FACE, 6, 1.0), (FACE2, 4, 1.0), (FACE, 4, 0.8), (FACE2, 3, 0.7)):
        for b in casc.detectMultiScale(g, 1.1, nb, minSize=(ms, ms)):
            boxes.append((tuple(int(v) for v in b), wgt))
    best, best_s = None, 0.0
    for (x, y, w, h), wgt in boxes:
        cxn, cyn, hn = (x + w / 2) / W, (y + h / 2) / H, h / H
        if cyn > 0.72 or hn > 0.75:
            continue
        eyes = EYE.detectMultiScale(g[y:y + int(h * 0.62), x:x + w], 1.1, 4, minSize=(max(8, w // 9),) * 2)
        eyes = [e for e in eyes if e[1] + e[3] / 2 < h * 0.55]
        if not eyes and skin is not None:
            patch = skin[y + int(0.2 * h):y + int(0.8 * h), x + int(0.2 * w):x + int(0.8 * w)]
            if patch.size and patch.mean() < 0.25:
                continue
        votes = sum(1 for (x2, y2, w2, h2), _ in boxes
                    if abs((x2 + w2 / 2) - (x + w / 2)) < 0.3 * w and abs((y2 + h2 / 2) - (y + h / 2)) < 0.3 * h)
        prior = np.exp(-((cxn - 0.5) / 0.28) ** 2) * np.exp(-((cyn - 0.38) / 0.25) ** 2)
        sc = hn * prior * wgt * (1 + min(len(eyes), 2)) * (1 + 0.25 * votes)
        if best is None or sc > best_s:
            best, best_s = ((x, y, w, h), eyes), sc
    return best


def _grab(img, x0, y0, x1, y1):
    a = np.asarray(img)
    H, W = a.shape[:2]
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
    out = cv2.copyMakeBorder(a[cy0:cy1, cx0:cx1], cy0 - y0, y1 - cy1, cx0 - x0, x1 - cx1, cv2.BORDER_REFLECT_101)
    return out


def locate_face(img):
    s = min(1.0, 900 / max(img.size))
    small = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR, reducing_gap=2.0)
    found = _find_face(np.asarray(small.convert("L")), np.asarray(small))
    return None if found is None else (found[0], found[1], s)
