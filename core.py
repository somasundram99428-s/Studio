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
def passport_crop(img, loc="auto", scale=1.0):
    if isinstance(loc, str):
        loc = locate_face(img)
    pw, ph = round(PH_W * scale), round(PH_H * scale)
    rs = Image.LANCZOS if scale >= 1 else Image.BICUBIC
    if loc is None:
        return enhance(ImageOps.fit(img, (pw, ph), rs, centering=(0.5, 0.4)), scale), False, False, False
    (x, y, w, h), eyes, s = loc
    cx, cy, fh = (x + w / 2) / s, (y + h / 2) / s, h / s
    ang = 0.0
    eyes = sorted(eyes, key=lambda e: -e[2])[:2]
    if len(eyes) == 2:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = sorted(eyes, key=lambda e: e[0])
        a = float(np.degrees(np.arctan2((y2 + h2 / 2) - (y1 + h1 / 2), (x2 + w2 / 2) - (x1 + w1 / 2))))
        if 0.7 < abs(a) < 12:
            ang = a

    ch = fh * CROP_H_FACTOR
    cw = ch * PH_W / PH_H
    if cw > img.width:
        f = max(0.85, img.width / cw)
        ch, cw = ch * f, cw * f
    top, left = cy - CROP_FACE_Y * ch, cx - cw / 2
    head_top = cy - 0.95 * fh
    if top + ch > img.height:
        top -= min(top + ch - img.height, max(0.0, head_top - 0.04 * ch - top))
    elif top < 0:
        top += min(-top, max(0.0, img.height - (top + ch)), max(0.0, head_top - 0.04 * ch - top))
    if cw <= img.width:
        left = min(max(left, 0.0), img.width - cw)
    bw, bh = max(8, int(round(cw))), max(8, int(round(ch)))
    x0, y0 = int(round(left)), int(round(top))

    pad = int(0.2 * ch) if ang else 0
    arr = _grab(img, x0 - pad, y0 - pad, x0 + bw + pad, y0 + bh + pad)
    ds = min(1.0, 2.0 * pw / bw) if (ang and scale < 1) else 1.0
    if ds < 0.8:
        arr = cv2.resize(arr, (max(1, round(arr.shape[1] * ds)), max(1, round(arr.shape[0] * ds))),
                         interpolation=cv2.INTER_AREA)
    else:
        ds = 1.0
    if ang:
        M = cv2.getRotationMatrix2D(((cx - (x0 - pad)) * ds, (cy - (y0 - pad)) * ds), ang, 1.0)
        arr = cv2.warpAffine(arr, M, (arr.shape[1], arr.shape[0]), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REFLECT_101)
    p2 = round(pad * ds)
    arr = arr[p2:p2 + round(bh * ds), p2:p2 + round(bw * ds)]
    low_res = bw < MIN_CROP_W
    result = Image.fromarray(arr).resize((pw, ph), rs)
    if low_res:
        result = result.filter(ImageFilter.UnsharpMask(max(0.5, 1.5 * scale), 80, 3))
    result = enhance(result, scale)
    bg_ok = False
    if CHANGE_BG:
        face_out = ((cx - x0) / bw * pw, (cy - y0) / bh * ph, fh / bh * ph)
        result, bg_ok = blue_background(result, face_out)
    return result, True, low_res, bg_ok


SHEETS = {4: (4, 1, 2, 2), 8: (4, 2, 2, 2)}


def make_sheet(photo, copies):
    cols, rows, cgap_mm, rgap_mm = SHEETS[copies]
    sheet = Image.new("RGB", (SH_W, SH_H), "white")
    cgap, rgap = mm(cgap_mm), mm(rgap_mm)
    cell_w, cell_h = photo.width, photo.height
    x0 = (SH_W - (cols * cell_w + (cols - 1) * cgap)) // 2
    y0 = (SH_H - (rows * cell_h + (rows - 1) * rgap)) // 2
    d = ImageDraw.Draw(sheet)
    for r in range(rows):
        for c in range(cols):
            x, y = x0 + c * (cell_w + cgap), y0 + r * (cell_h + rgap)
            sheet.paste(photo, (x, y))
            d.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline=(0, 0, 0), width=BORDER)
    return sheet


FULL_MARGIN = mm(4)


def make_full(img):
    if img.height > img.width:
        img = img.rotate(90, expand=True)
    inner_w, inner_h = SH_W - 2 * FULL_MARGIN, SH_H - 2 * FULL_MARGIN
    photo = enhance(ImageOps.fit(img, (inner_w - 2 * BORDER, inner_h - 2 * BORDER), Image.LANCZOS))
    sheet = Image.new("RGB", (SH_W, SH_H), "white")
    x, y = FULL_MARGIN, FULL_MARGIN
    sheet.paste(photo, (x + BORDER, y + BORDER))
    ImageDraw.Draw(sheet).rectangle((x, y, x + inner_w - 1, y + inner_h - 1), outline=(0, 0, 0), width=BORDER)
    return sheet
