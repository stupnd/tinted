"""Illuminant correction and contrast normalisation utilities.

These preprocessing steps run before face detection so that downstream
LAB distance measurements work against a lighting-corrected image rather
than raw sensor values.

Public API
----------
preprocess_image(img_bytes)         -> np.ndarray  (decode + white balance)
neutral_pixel_whitebalance(img_bgr) -> np.ndarray  (illuminant correction used by preprocess)
gray_world_whitebalance(img_bgr)    -> np.ndarray  (whole-image illuminant correction)
apply_clahe(img_bgr)                -> np.ndarray  (contrast boost — detection only)
"""

from __future__ import annotations

import cv2
import numpy as np


def gray_world_whitebalance(img_bgr: np.ndarray) -> np.ndarray:
    """Scale each BGR channel so its mean equals the grand mean across all
    three channels (classic gray-world illuminant assumption).

    The assumption is that, averaged across the whole image, scene colour is
    neutral.  Under that assumption any per-channel deviation from the grand
    mean must be due to the illuminant, and dividing it out removes the cast.

    This is a parameter-free, O(pixels) first-order correction that handles
    the most common failure mode: warm incandescent / ring-light tints
    pushing fair skin toward MST-5 territory.

    Output is clamped to [0, 255] uint8.
    """
    img = img_bgr.astype(np.float32)

    mean_b = float(img[:, :, 0].mean())
    mean_g = float(img[:, :, 1].mean())
    mean_r = float(img[:, :, 2].mean())
    grand_mean = (mean_b + mean_g + mean_r) / 3.0

    # Scale each channel; skip channels that are essentially black to avoid
    # division by zero on pathological (e.g. solid-black) images.
    for ch, ch_mean in enumerate((mean_b, mean_g, mean_r)):
        if ch_mean > 1e-3:
            img[:, :, ch] *= grand_mean / ch_mean

    return np.clip(img, 0, 255).astype(np.uint8)


# Pixels count as a neutral reference when HSV saturation is below this (0-255)
# and they are neither in deep shadow nor clipped.
_NEUTRAL_S_MAX = 40
_NEUTRAL_V_MIN = 30
_NEUTRAL_V_MAX = 245
# Below this fraction of neutral pixels there is nothing trustworthy to
# balance against, so the image is left as shot.
_NEUTRAL_MIN_FRAC = 0.05
# A real illuminant cast rarely needs more than this per-channel correction;
# larger gains mean the "neutral" pixels weren't neutral.
_MAX_CHANNEL_GAIN = 1.25


def neutral_pixel_whitebalance(img_bgr: np.ndarray) -> np.ndarray:
    """Gray-world correction estimated only from near-neutral pixels.

    Plain gray-world assumes the whole frame averages to grey. In a close-up
    selfie the frame is mostly skin, so it treats the skin itself as the cast
    and pulls it to grey (a medium-warm #ca8263 came out as #949295). Here the
    illuminant is measured from low-saturation pixels only — walls, clothing,
    the whites of the eyes — and skin never votes on its own correction.

    When fewer than ``_NEUTRAL_MIN_FRAC`` of pixels are neutral the image is
    returned unchanged: an uncorrected photo is a smaller error than a guessed
    correction. Gains are clamped to ``_MAX_CHANNEL_GAIN`` either way.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    neutral = (
        (hsv[:, :, 1] < _NEUTRAL_S_MAX)
        & (hsv[:, :, 2] > _NEUTRAL_V_MIN)
        & (hsv[:, :, 2] < _NEUTRAL_V_MAX)
    )
    if neutral.mean() < _NEUTRAL_MIN_FRAC:
        return img_bgr

    means = img_bgr[neutral].astype(np.float32).mean(axis=0)
    gains = np.clip(
        means.mean() / np.maximum(means, 1e-3),
        1.0 / _MAX_CHANNEL_GAIN,
        _MAX_CHANNEL_GAIN,
    )
    return np.clip(img_bgr.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L* channel only, leaving hue and saturation intact.

    Converting to LAB, equalising L*, then converting back means that local
    contrast is improved without shifting the colour that skin-tone extraction
    depends on.

    clipLimit=2.0 and tileGridSize=(8,8) are the conventional defaults for
    facial preprocessing — aggressive enough to recover shadow detail, mild
    enough not to amplify noise in flat skin regions.

    Returns a BGR uint8 image.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_image(img_bytes: bytes) -> np.ndarray:
    """Decode *img_bytes* to a BGR ndarray and apply neutral-pixel white balance.

    CLAHE is deliberately NOT applied here: it redistributes luminance and
    would bias the skin-colour measurement (~1 MST level deeper).  Landmark
    detection applies CLAHE to its own working copy instead — see
    ``detection.face_detection._detect_landmarks``.

    Returns a colour-faithful BGR uint8 image ready for skin sampling.

    Raises:
        ValueError: if the bytes cannot be decoded as a valid image.
    """
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(
            "Could not decode the image — "
            "please upload a valid JPEG, PNG, or WebP file."
        )

    return neutral_pixel_whitebalance(img_bgr)
