import cv2
import numpy as np
import time


#  NOISE DETECTION  (image-level, unchanged)

def detect_noise_type(img_gray):
    """Classify dominant noise type of the whole image."""
    if len(img_gray.shape) != 2:
        raise ValueError("detect_noise_type expects a grayscale image")

    h, w  = img_gray.shape
    total = h * w
    img   = img_gray

    sp_ratio = (np.sum(img == 0) + np.sum(img == 255)) / total

    img_f    = img.astype(np.float32)
    mean     = cv2.blur(img_f, (7, 7))
    mean_sq  = cv2.blur(img_f * img_f, (7, 7))
    var      = np.maximum(mean_sq - mean * mean, 0)
    cv_value = np.mean(np.sqrt(var) / (mean + 1e-3))

    if sp_ratio > 0.02:
        return "salt_pepper"
    elif cv_value > 0.5:
        return "speckle"
    else:
        return "gaussian"



#  PIXEL-LEVEL ADAPTIVE ANALYSIS


def compute_pixel_noise_maps(img_gray, patch=7):
    """
    For every pixel compute three local statistics:
      • sp_flag   – likely salt-or-pepper outlier  (bool map)
      • cv_map    – coefficient of variation        (float map, speckle indicator)
      • std_map   – local std deviation             (float map, gaussian indicator)
    Returns all three maps (same H×W as input).
    """
    half = patch // 2
    img_f = img_gray.astype(np.float32)

    # Pad so every pixel gets a full patch
    padded = cv2.copyMakeBorder(img_f, half, half, half, half, cv2.BORDER_REFLECT)

    mean   = cv2.blur(padded, (patch, patch))
    mean_sq= cv2.blur(padded * padded, (patch, patch))
    var    = np.maximum(mean_sq - mean * mean, 0)
    std    = np.sqrt(var)
    cv_map = std / (mean + 1e-3)

    # Crop back to original size
    mean   = mean  [half:-half, half:-half]
    std    = std   [half:-half, half:-half]
    cv_map = cv_map[half:-half, half:-half]

    # Salt-pepper: pixel deviates far from its local mean
    diff      = np.abs(img_f - mean)
    sp_flag   = (diff > 3.5 * std + 10) & ((img_gray == 0) | (img_gray == 255))

    return sp_flag.astype(bool), cv_map, std


# ─────────────────────────────────────────────
#  PER-PIXEL FILTER SELECTOR
# ─────────────────────────────────────────────

def adaptive_pixel_filter(img_bgr, patch=7):
    """
    Analyse every pixel and apply the most appropriate filter locally:

      salt-pepper pixel  → replace with local median
      speckle pixel      → replace with homomorphic (log-domain) Gaussian smooth
      gaussian pixel     → replace with NLM-based value  (pre-computed)
      clean pixel        → keep original value

    Returns the restored BGR image and a label map (0=clean,1=sp,2=speckle,3=gauss).
    """
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sp_flag, cv_map, std_map = compute_pixel_noise_maps(gray, patch=patch)

    # Pre-compute three global-filter outputs (cheap fallback layers)
    median_bgr  = cv2.medianBlur(img_bgr, patch | 1)          # salt-pepper
    gauss_bgr   = cv2.fastNlMeansDenoisingColored(             # gaussian
                      img_bgr, None, h=10, hColor=10,
                      templateWindowSize=7, searchWindowSize=21)

    # Homomorphic smooth for speckle
    def _homomorphic_channel(ch):
        f = ch.astype(np.float32)
        f[f <= 0] = 1.0
        return np.clip(np.exp(cv2.GaussianBlur(np.log(f), (5, 5), 1.0)), 0, 255).astype(np.uint8)

    speckle_bgr = cv2.merge([_homomorphic_channel(c)
                              for c in cv2.split(img_bgr)])

    # ── Per-pixel decision ──────────────────────────────────────────
    # Priority: salt-pepper  >  speckle  >  gaussian  >  clean
    SPECKLE_CV  = 0.50
    GAUSS_STD   = 8.0

    speckle_flag = (~sp_flag) & (cv_map  > SPECKLE_CV)
    gauss_flag   = (~sp_flag) & (~speckle_flag) & (std_map > GAUSS_STD)

    # Label map for inspection (0=clean,1=sp,2=speckle,3=gauss)
    label_map = np.zeros(gray.shape, dtype=np.uint8)
    label_map[sp_flag]      = 1
    label_map[speckle_flag] = 2
    label_map[gauss_flag]   = 3

    # Expand flags to 3-channel masks
    def _mask3(flag):
        return np.stack([flag]*3, axis=-1)

    restored = img_bgr.copy()
    restored = np.where(_mask3(sp_flag),      median_bgr,  restored)
    restored = np.where(_mask3(speckle_flag), speckle_bgr, restored)
    restored = np.where(_mask3(gauss_flag),   gauss_bgr,   restored)

    return restored.astype(np.uint8), label_map



#  IMAGE-LEVEL AUTO DENOISE  (original logic)


def auto_denoise(noisy_bgr):
    gray       = cv2.cvtColor(noisy_bgr, cv2.COLOR_BGR2GRAY)
    noise_type = detect_noise_type(gray)

    if   noise_type == "salt_pepper":
        restored = cv2.medianBlur(cv2.medianBlur(noisy_bgr, 3), 3)
    elif noise_type == "gaussian":
        restored = cv2.fastNlMeansDenoisingColored(noisy_bgr, None,
                       h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    elif noise_type == "speckle":
        def _homo(ch):
            f = ch.astype(np.float32); f[f<=0] = 1.0
            return np.clip(np.exp(cv2.GaussianBlur(np.log(f),(5,5),1.0)),0,255).astype(np.uint8)
        restored = cv2.merge([_homo(c) for c in cv2.split(noisy_bgr)])
    else:
        restored = noisy_bgr.copy()

    return restored, noise_type



#  MAIN


if __name__ == "__main__":
    import sys

    noisy_path = sys.argv[1] if len(sys.argv) > 1 else "noisy_image.png"

    noisy = cv2.imread(noisy_path)
    if noisy is None:
        raise FileNotFoundError(f"Could not open: {noisy_path}")
    if noisy.ndim < 3:
        noisy = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)

    print(f"Image size : {noisy.shape[1]}×{noisy.shape[0]}  ({noisy.shape[1]*noisy.shape[0]:,} pixels)")

    # ── 1. Image-level auto denoise ─────────────────────────────────
    t0 = time.perf_counter()
    restored_auto, noise_type = auto_denoise(noisy)
    t_auto = time.perf_counter() - t0
    print(f"\n[Image-level denoising]")
    print(f"  Detected noise type : {noise_type}")
    print(f"  CPU time            : {t_auto*1000:.2f} ms")

    # ── 2. Pixel-by-pixel adaptive filter ───────────────────────────
    print(f"\n[Pixel-by-pixel adaptive filter]")
    t1 = time.perf_counter()
    restored_adaptive, label_map = adaptive_pixel_filter(noisy, patch=7)
    t_adaptive = time.perf_counter() - t1
    print(f"  CPU time            : {t_adaptive*1000:.2f} ms")

    h, w = label_map.shape
    total = h * w
    n_sp      = int(np.sum(label_map == 1))
    n_speckle = int(np.sum(label_map == 2))
    n_gauss   = int(np.sum(label_map == 3))
    n_clean   = total - n_sp - n_speckle - n_gauss
    print(f"  Pixel breakdown     :")
    print(f"    Clean   : {n_clean:>9,}  ({100*n_clean/total:5.1f}%)")
    print(f"    S&P     : {n_sp:>9,}  ({100*n_sp/total:5.1f}%)")
    print(f"    Speckle : {n_speckle:>9,}  ({100*n_speckle/total:5.1f}%)")
    print(f"    Gaussian: {n_gauss:>9,}  ({100*n_gauss/total:5.1f}%)")

    # ── 3. Colour-coded label map 
    # 0=clean(black) 1=S&P(red) 2=speckle(green) 3=gaussian(blue)
    colour_map = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    colour_map[label_map == 1] = (0,   0,   255)   # red   – salt-pepper
    colour_map[label_map == 2] = (0,   200, 0  )   # green – speckle
    colour_map[label_map == 3] = (255, 100, 0  )   # blue  – gaussian

    # ── Save outputs 
    cv2.imwrite("restored_image_level.png",  restored_auto)
    cv2.imwrite("restored_pixel_level.png",  restored_adaptive)
    cv2.imwrite("noise_label_map.png",       colour_map)
    print("\n[Saved]")
    print("  restored_image_level.png  – classic auto-denoise result")
    print("  restored_pixel_level.png  – pixel-by-pixel adaptive result")
    print("  noise_label_map.png       – red=S&P  green=speckle  blue=gaussian")

    # ── Display
    cv2.imshow("Noisy (original)",            noisy)
    cv2.imshow("Restored – image-level",      restored_auto)
    cv2.imshow("Restored – pixel adaptive",   restored_adaptive)
    cv2.imshow("Noise label map",             colour_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()