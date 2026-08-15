import cv2
import numpy as np
import time


def detect_noise_type(img_gray):

    h, w = img_gray.shape
    total = h * w

    sp_ratio = np.sum(
        (img_gray <= 10) | (img_gray >= 245)
    ) / total

    img_f = img_gray.astype(np.float32)

    mean = cv2.blur(img_f, (7, 7))

    mean_sq = cv2.blur(img_f * img_f, (7, 7))

    var = np.maximum(mean_sq - mean * mean, 0)

    cv_value = np.mean(
        np.sqrt(var) / (mean + 1e-3)
    )

    if sp_ratio > 0.02:
        return "salt_pepper"
    elif cv_value > 0.7:
        return "speckle"
    else:
        return "gaussian"




def compute_pixel_noise_maps(img_gray, patch=7):

    img_f = img_gray.astype(np.float32)

    mean = cv2.blur(img_f, (patch, patch))

    mean_sq = cv2.blur(img_f * img_f, (patch, patch))

    var = np.maximum(mean_sq - mean * mean, 0)

    std = np.sqrt(var)

    cv_map = std / (mean + 1e-3)

    median = cv2.medianBlur(img_gray, patch)

    diff_median = np.abs(
        img_f - median.astype(np.float32)
    )

    
    sp_flag = (
        ((img_gray <= 15) | (img_gray >= 240))
        &
        (diff_median > 25)
    )

    return sp_flag, cv_map, std




def homomorphic_channel(ch):

    f = ch.astype(np.float32)

    f[f <= 0] = 1.0

    log_img = np.log(f)

    smooth = cv2.GaussianBlur(
        log_img,
        (5, 5),
        1.0
    )

    result = np.exp(smooth)

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)




def adaptive_pixel_filter(img_bgr, patch=7):

    gray = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2GRAY
    )

    sp_flag, cv_map, std_map = compute_pixel_noise_maps(
        gray,
        patch
    )

    

    median_bgr = cv2.medianBlur(
        img_bgr,
        5
    )

    gaussian_bgr = cv2.fastNlMeansDenoisingColored(
        img_bgr,
        None,
        h=10,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    speckle_bgr = cv2.merge([
        homomorphic_channel(c)
        for c in cv2.split(img_bgr)
    ])

    
    SPECKLE_CV = 0.90
    GAUSS_STD = 20

    speckle_flag = (
        (~sp_flag)
        &
        (cv_map > SPECKLE_CV)
    )

    gauss_flag = (
        (~sp_flag)
        &
        (~speckle_flag)
        &
        (std_map > GAUSS_STD)
        &
        (std_map < 60)
    )

    

    label_map = np.zeros(
        gray.shape,
        dtype=np.uint8
    )

    label_map[sp_flag] = 1
    label_map[speckle_flag] = 2
    label_map[gauss_flag] = 3

    

    restored = img_bgr.copy()

    mask_sp = np.stack(
        [sp_flag] * 3,
        axis=-1
    )

    mask_speckle = np.stack(
        [speckle_flag] * 3,
        axis=-1
    )

    mask_gauss = np.stack(
        [gauss_flag] * 3,
        axis=-1
    )

    restored = np.where(
        mask_sp,
        median_bgr,
        restored
    )

    restored = np.where(
        mask_speckle,
        speckle_bgr,
        restored
    )

    restored = np.where(
        mask_gauss,
        gaussian_bgr,
        restored
    )

    return restored.astype(np.uint8), label_map




def auto_denoise(noisy_bgr):

    gray = cv2.cvtColor(
        noisy_bgr,
        cv2.COLOR_BGR2GRAY
    )

    noise_type = detect_noise_type(gray)

    if noise_type == "salt_pepper":

        restored = cv2.medianBlur(
            cv2.medianBlur(noisy_bgr, 3),
            3
        )

    elif noise_type == "gaussian":

        restored = cv2.fastNlMeansDenoisingColored(
            noisy_bgr,
            None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )

    elif noise_type == "speckle":

        restored = cv2.merge([
            homomorphic_channel(c)
            for c in cv2.split(noisy_bgr)
        ])

    else:

        restored = noisy_bgr.copy()

    return restored, noise_type


if __name__ == "__main__":

    noisy_path = "noisy_image.png"

    noisy = cv2.imread(noisy_path)

    if noisy is None:
        raise FileNotFoundError(
            f"Cannot open {noisy_path}"
        )

    print(
        f"Image size : "
        f"{noisy.shape[1]}x{noisy.shape[0]}"
    )

   

    start = time.perf_counter()

    restored_auto, noise_type = auto_denoise(noisy)

    t1 = time.perf_counter() - start

    print("\nIMAGE LEVEL")

    print(
        f"Detected noise : {noise_type}"
    )

    print(
        f"CPU time : {t1*1000:.2f} ms"
    )

   

    start = time.perf_counter()

    restored_adaptive, label_map = adaptive_pixel_filter(
        noisy,
        patch=7
    )

    t2 = time.perf_counter() - start

    print("\nPIXEL LEVEL")

    print(
        f"CPU time : {t2*1000:.2f} ms"
    )

    total = label_map.size

    n_sp = np.sum(label_map == 1)

    n_speckle = np.sum(label_map == 2)

    n_gauss = np.sum(label_map == 3)

    n_clean = total - n_sp - n_speckle - n_gauss

    print("\nPixel Breakdown")

    print(
        f"Clean    : {n_clean:,}"
        f" ({100*n_clean/total:.1f}%)"
    )

    print(
        f"S&P      : {n_sp:,}"
        f" ({100*n_sp/total:.1f}%)"
    )

    print(
        f"Speckle  : {n_speckle:,}"
        f" ({100*n_speckle/total:.1f}%)"
    )

    print(
        f"Gaussian : {n_gauss:,}"
        f" ({100*n_gauss/total:.1f}%)"
    )


    colour_map = np.zeros(
        (*label_map.shape, 3),
        dtype=np.uint8
    )

    colour_map[label_map == 1] = (0, 0, 255)
    colour_map[label_map == 2] = (0, 255, 0)
    colour_map[label_map == 3] = (255, 0, 0)


    cv2.imwrite(
        "restored_image_level.png",
        restored_auto
    )

    cv2.imwrite(
        "restored_pixel_level.png",
        restored_adaptive
    )

    cv2.imwrite(
        "noise_label_map.png",
        colour_map
    )

    print("\nSaved Successfully")

    cv2.imshow("Original", noisy)
    cv2.imshow("Image Level", restored_auto)
    cv2.imshow("Pixel Adaptive", restored_adaptive)
    cv2.imshow("Noise Map", colour_map)

    cv2.waitKey(0)
    cv2.destroyAllWindows()