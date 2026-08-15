import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim



# Add Mixed Noise (Gaussian + Salt & Pepper)

def add_mixed_noise(image, gauss_var=20, sp_amount=0.05):
    noisy = image.astype(np.float32)

    # Gaussian noise
    gauss = np.random.normal(0, gauss_var ** 0.5, image.shape)
    noisy += gauss

    # Salt & Pepper noise
    s_vs_p = 0.5
    out = noisy.copy()
    num_salt = int(sp_amount * image.size * s_vs_p)
    num_pepper = int(sp_amount * image.size * (1 - s_vs_p))

    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    out[tuple(coords)] = 255

    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    out[tuple(coords)] = 0

    return np.clip(out, 0, 255).astype(np.uint8)



# Image Enhancement Factor (IEF)

def compute_IEF(original, noisy, restored):
    num = np.sum((noisy - original) ** 2)
    den = np.sum((restored - original) ** 2)
    return num / den if den != 0 else float('inf')



# Adaptive Hybrid Filter

def adaptive_hybrid_filter(image, gaussian_ksize=5, median_ksize=3):
    image = image.astype(np.float32)

    gaussian_filtered = cv2.GaussianBlur(image, (5, 5), 0)
    median_filtered = cv2.medianBlur(image.astype(np.uint8), 3).astype(np.float32)

    mean = cv2.blur(image, (3, 3))
    mean_sq = cv2.blur(image ** 2, (3, 3))
    local_var = mean_sq - mean ** 2

    alpha = cv2.normalize(local_var, None, 0, 1, cv2.NORM_MINMAX)
    hybrid = alpha * gaussian_filtered + (1 - alpha) * median_filtered

    return np.clip(hybrid, 0, 255).astype(np.uint8)



# MAIN

if __name__ == "__main__":

    original = cv2.imread("original_image.png", cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError("original_image.png not found")

    # Noise levels: 5%, 15%, 20%
    noise_levels = [0.05, 0.15, 0.20]

    print("\n===== PSNR, SSIM, IEF FOR DIFFERENT NOISE LEVELS =====")

    for sp in noise_levels:
        noisy = add_mixed_noise(original, gauss_var=25, sp_amount=sp)
        restored = adaptive_hybrid_filter(noisy)

        psnr_noisy = psnr(original, noisy)
        psnr_restored = psnr(original, restored)

        ssim_noisy = ssim(original, noisy, data_range=255)
        ssim_restored = ssim(original, restored, data_range=255)

        ief_value = compute_IEF(original, noisy, restored)

        print(f"\n--- Noise Level: {int(sp*100)}% ---")
        print(f"PSNR (Before):  {psnr_noisy:.4f} dB")
        print(f"PSNR (After):   {psnr_restored:.4f} dB")
        print(f"SSIM (Before):  {ssim_noisy:.4f}")
        print(f"SSIM (After):   {ssim_restored:.4f}")
        print(f"IEF:            {ief_value:.4f}")

        # Save images
        cv2.imwrite(f"noisy_{int(sp*100)}.png", noisy)
        cv2.imwrite(f"restored_{int(sp*100)}.png", restored)

    print("\n[INFO] All noisy and restored images .")


