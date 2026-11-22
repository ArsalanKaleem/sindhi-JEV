"""Make a clean render look like a scan / photocopy / phone photo.

Ported from Sindhi-Benchmark/ocr/render/degrade.py, plus low resolution,
shear/skew, speckle and a severity knob.

Label safety: the degradation must not erase or merge dots, or a "correct"
candidate would no longer match the scan. So severity is capped (default
max 0.8) and speckles are small single blobs; scripts/check_degrade.py
measures how often a dot-swap stays visible after degradation.
"""
import random

import cv2
import numpy as np


def _background(h, w, rng, nprng):
    small = rng.uniform(200, 255) - nprng.random((4, 4)) * rng.uniform(0, 40)
    return np.clip(cv2.resize(small.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC), 0, 255)


def degrade(img, rng=None, severity=None, max_severity=0.8, realism=False):
    """img: uint8 HxW, black text on white. Returns (uint8 image, params dict).
    realism=True (data_v3) adds uneven lighting and occasional binarisation; False keeps v1/v2 output identical."""
    rng = rng or random.Random()
    nprng = np.random.default_rng(rng.randrange(1 << 30))
    s = rng.uniform(0.1, max_severity) if severity is None else severity
    p = {"severity": round(s, 3)}
    img = img.astype(np.float32)
    h, w = img.shape

    if rng.random() < 0.5:  # skew / shear (page not flat on scanner)
        ang = rng.uniform(-2.5, 2.5) * s
        sh = rng.uniform(-0.08, 0.08) * s
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        M[0, 1] += sh
        img = cv2.warpAffine(img, M, (w, h), borderValue=255, flags=cv2.INTER_LINEAR)
        p["rot"], p["shear"] = round(ang, 2), round(sh, 3)
    if rng.random() < 0.4:  # paper texture
        img = np.minimum(img, _background(h, w, rng, nprng))
        p["texture"] = 1
    if rng.random() < 0.5:  # thin or thick ink (only at low severity: erosion can drop dots)
        k = np.ones((2, 2), np.uint8)
        if rng.random() < 0.5 and s < 0.6:
            img = cv2.dilate(img, k)  # dilate white = thinner ink
            p["ink"] = "thin"
        else:
            img = cv2.erode(img, k)  # erode white = thicker ink
            p["ink"] = "thick"
    if rng.random() < 0.3:  # faded ink
        f = rng.uniform(0.85 - 0.35 * s, 0.9)
        img = 255 - (255 - img) * f
        p["fade"] = round(f, 2)
    if rng.random() < 0.6:  # low resolution: down then up
        f = rng.uniform(1.0 - 0.6 * s, 1.0)
        small = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        p["scale"] = round(f, 2)
    if rng.random() < 0.7:
        sig = rng.uniform(0.3, 0.3 + 1.0 * s)
        img = cv2.GaussianBlur(img, (0, 0), sig)
        p["blur"] = round(sig, 2)
    if rng.random() < 0.4:  # speckles: small dark specks that look like stray dots
        n = int(rng.uniform(0, 6) * s * w * h / 20000) + 1
        for _ in range(n):
            cv2.circle(img, (rng.randrange(w), rng.randrange(h)), rng.choice([0, 1]), rng.uniform(40, 160), -1)
        p["speckles"] = n
    if rng.random() < 0.6:
        sd = rng.uniform(3, 4 + 14 * s)
        img = img + nprng.normal(0, sd, img.shape)
        p["noise"] = round(sd, 1)
    if realism and rng.random() < 0.3:  # uneven lighting (scanner lid, phone photo)
        gx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
        gy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        a, bb = rng.uniform(-1, 1), rng.uniform(-1, 1)
        g = 1.0 - rng.uniform(0.1, 0.35) * np.clip(a * gx + bb * gy + rng.uniform(0, 0.5), 0, 1)
        img = img * g
        p["light"] = 1
    img = np.clip(img, 0, 255).astype(np.uint8)
    if realism and rng.random() < 0.15:  # binarised scan (common in old digitised books)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        p["binary"] = 1
    q = rng.randint(int(90 - 55 * s), 95)
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    p["jpeg"] = q
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE), p
