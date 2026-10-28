"""Unit tests for color_utils preprocessing functions."""

import numpy as np
import pytest

from color_utils import (
    apply_clahe,
    gray_world_whitebalance,
    neutral_pixel_whitebalance,
    preprocess_image,
)


# ---------------------------------------------------------------------------
# gray_world_whitebalance
# ---------------------------------------------------------------------------

def test_gray_world_corrects_warm_tint():
    """A neutral gray patch boosted +30 on the R channel (warm tint) should
    be corrected so the R mean is within 5 of the global per-channel mean."""
    neutral = 120
    img = np.full((100, 100, 3), neutral, dtype=np.uint8)
    img[:, :, 2] = min(neutral + 30, 255)  # R is channel index 2 in BGR

    result = gray_world_whitebalance(img)

    mean_b = float(result[:, :, 0].mean())
    mean_g = float(result[:, :, 1].mean())
    mean_r = float(result[:, :, 2].mean())
    global_mean = (mean_b + mean_g + mean_r) / 3.0

    assert abs(mean_r - global_mean) < 5, (
        f"After white balance: R mean {mean_r:.1f} should be within 5 of "
        f"global mean {global_mean:.1f}"
    )


def test_gray_world_neutral_image_is_unchanged():
    """A perfectly neutral image (B == G == R) should pass through unchanged."""
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    result = gray_world_whitebalance(img)
    np.testing.assert_array_equal(result, img)


def test_gray_world_output_dtype_and_range():
    """Output must be uint8 with all values in [0, 255]."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    result = gray_world_whitebalance(img)
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


def test_gray_world_cool_tint_corrected():
    """A blue-channel bias (cool tint) should be corrected symmetrically."""
    neutral = 100
    img = np.full((60, 60, 3), neutral, dtype=np.uint8)
    img[:, :, 0] = min(neutral + 40, 255)  # B is channel index 0

    result = gray_world_whitebalance(img)

    mean_b = float(result[:, :, 0].mean())
    mean_g = float(result[:, :, 1].mean())
    mean_r = float(result[:, :, 2].mean())
    global_mean = (mean_b + mean_g + mean_r) / 3.0

    assert abs(mean_b - global_mean) < 5


# ---------------------------------------------------------------------------
# apply_clahe
# ---------------------------------------------------------------------------

def test_clahe_preserves_shape_and_dtype():
    """CLAHE must not change image dimensions or dtype."""
    img = np.full((80, 80, 3), 100, dtype=np.uint8)
    result = apply_clahe(img)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_clahe_does_not_shift_hue_on_uniform_image():
    """On a spatially uniform image CLAHE should not alter the per-pixel hue.

    Uniform images have no local contrast to equalise, so the L channel
    should emerge at roughly the same value and hue channels are untouched.
    """
    import cv2

    img = np.full((80, 80, 3), 120, dtype=np.uint8)
    result = apply_clahe(img)

    # Convert both to HSV to compare hue channel directly.
    hue_before = cv2.cvtColor(img,    cv2.COLOR_BGR2HSV)[:, :, 0]
    hue_after  = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)[:, :, 0]
    np.testing.assert_array_equal(hue_before, hue_after)


# neutral_pixel_whitebalance

def test_neutral_wb_leaves_skin_dominated_image_alone():
    # A frame that is all skin has no neutral reference; plain gray-world
    # would drag it to grey.
    img = np.full((100, 100, 3), (99, 130, 202), dtype=np.uint8)  # BGR, #ca8263
    np.testing.assert_array_equal(neutral_pixel_whitebalance(img), img)


def test_neutral_wb_corrects_cast_measured_on_neutral_background():
    # Skin patch on a grey wall under slightly warm light: the wall reads (118,122,130).
    img = np.full((100, 100, 3), (118, 122, 130), dtype=np.uint8)
    img[30:70, 30:70] = (99, 130, 202)
    result = neutral_pixel_whitebalance(img).astype(np.float32)
    wall = result[0:10, 0:10].reshape(-1, 3).mean(axis=0)
    assert wall.max() - wall.min() < 3
    # Skin keeps its hue: still clearly red > green > blue.
    b, g, r = result[50, 50]
    assert r > g > b


def test_neutral_wb_caps_correction_gain():
    img = np.full((100, 100, 3), (60, 90, 95), dtype=np.uint8)  # low-sat, strong cast
    result = neutral_pixel_whitebalance(img).astype(np.float32)
    assert result[0, 0, 0] <= 60 * 1.25 + 1


# ---------------------------------------------------------------------------
# preprocess_image
# ---------------------------------------------------------------------------

def test_preprocess_image_invalid_bytes_raises():
    """Garbage bytes should raise ValueError, not crash with an unrelated error."""
    with pytest.raises(ValueError, match="Could not decode"):
        preprocess_image(b"not an image")


def test_preprocess_image_returns_bgr_ndarray():
    """A valid JPEG round-trip should return a 3-channel uint8 ndarray."""
    import cv2

    img = np.full((40, 40, 3), 150, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    result = preprocess_image(buf.tobytes())
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3
    assert result.shape[2] == 3
    assert result.dtype == np.uint8
