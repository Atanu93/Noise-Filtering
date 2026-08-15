import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

img = cv2.imread("original_image.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def add_gaussian_noise(image, mean=0, var=20):
    sigma = var ** 0.7
    gaussian = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + gaussian
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image, prob=0.07):
    noisy = np.copy(image)
    rnd = np.random.rand(image.shape[0], image.shape[1])
    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 255
    return noisy

def add_speckle_noise(image):
    noise = np.random.randn(*image.shape)
    noisy = image + image * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

noises = {
    "Gaussian": add_gaussian_noise(img),
    "Salt & Pepper": add_salt_pepper_noise(img),
    "Speckle": add_speckle_noise(img)
}

def apply_mean_filter(image, kernel_size=3):
    return cv2.blur(image, (kernel_size, kernel_size))

def adaptive_mean_filter(image, kernel_size=3):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    local_mean = cv2.blur(gray, (kernel_size, kernel_size))
    filtered = cv2.subtract(gray, local_mean)
    filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)

def apply_median_filter(image, kernel_size=3):
    filtered = np.zeros_like(image)
    for i in range(3):
        filtered[:, :, i] = cv2.medianBlur(image[:, :, i], kernel_size)
    return filtered

def apply_gaussian_filter(image, kernel_size=5, sigma=1):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def adaptive_gaussian_filter(image, noise_type):
    sigma = {"gaussian": 1, "salt_pepper": 2, "speckle": 1.5}.get(noise_type, 1)
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

filters = {
    "Mean Filter": apply_mean_filter,
    "Adaptive Mean Filter": adaptive_mean_filter,
    "Median Filter": apply_median_filter,
    "Gaussian Filter": apply_gaussian_filter,
    "Adaptive Gaussian Filter": adaptive_gaussian_filter
}

def calculate_ief(original, noisy, denoised):
    numerator = np.sum((noisy.astype(np.float32) - original.astype(np.float32)) ** 2)
    denominator = np.sum((denoised.astype(np.float32) - original.astype(np.float32)) ** 2)
    return float('inf') if denominator == 0 else numerator / denominator

for noise_name, noisy_img in noises.items():
    print(f"\n--- Metrics for Noise Type: {noise_name} ---")

    plt.figure(figsize=(20, 8))

    plt.subplot(2, 4, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.imshow(noisy_img)
    plt.title(f"{noise_name} Noise")
    plt.axis('off')

    filter_idx = 3
    for filter_name, filter_func in filters.items():
        if filter_name == "Adaptive Gaussian Filter":
            denoised = filter_func(noisy_img, noise_name.lower())
        else:
            denoised = filter_func(noisy_img)

        plt.subplot(2, 4, filter_idx)
        plt.imshow(denoised)
        plt.title(filter_name)
        plt.axis('off')
        filter_idx += 1

        print(f"{filter_name} → PSNR: {psnr(img, denoised):.4f} | SSIM: {ssim(img, denoised, channel_axis=2):.4f} | IEF: {calculate_ief(img, noisy_img, denoised):.4f}")

    plt.show()


