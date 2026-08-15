import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

img = cv2.imread("original_image.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def add_gaussian_noise(image, mean=0, var=20):
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + gaussian
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def add_salt_pepper_noise(image, prob=0.02):
    noisy = np.copy(image)
    rnd = np.random.rand(image.shape[0], image.shape[1])
    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 255
    return noisy

def add_speckle_noise(image):
    noise = np.random.randn(*image.shape)
    noisy = image + image * noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

gaussian_img = add_gaussian_noise(img)
s_p_img = add_salt_pepper_noise(img)
speckle_img = add_speckle_noise(img)


def apply_gaussian_filter(image, kernel_size=5, sigma=1):
    filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return filtered

gaussian_denoised = apply_gaussian_filter(gaussian_img)
s_p_denoised = apply_gaussian_filter(s_p_img)
speckle_denoised = apply_gaussian_filter(speckle_img)

def calculate_ief(original, noisy, denoised):
    numerator = np.sum((noisy.astype(np.float32) - original.astype(np.float32)) ** 2)
    denominator = np.sum((denoised.astype(np.float32) - original.astype(np.float32)) ** 2)
    if denominator == 0:
        return float('inf')
    return numerator / denominator

print("\n Quality Metrics Results (with IEF):")
print("Gaussian Noise → PSNR:", psnr(img, gaussian_denoised), 
      "| SSIM:", ssim(img, gaussian_denoised, channel_axis=2), 
      "| IEF:", calculate_ief(img, gaussian_img, gaussian_denoised))

print("Salt & Pepper → PSNR:", psnr(img, s_p_denoised), 
      "| SSIM:", ssim(img, s_p_denoised, channel_axis=2), 
      "| IEF:", calculate_ief(img, s_p_img, s_p_denoised))

print("Speckle Noise → PSNR:", psnr(img, speckle_denoised), 
      "| SSIM:", ssim(img, speckle_denoised, channel_axis=2), 
      "| IEF:", calculate_ief(img, speckle_img, speckle_denoised))


plt.figure(figsize=(15, 10))

plt.subplot(3, 3, 1), plt.imshow(img), plt.title("Original"), plt.axis('off')
plt.subplot(3, 3, 2), plt.imshow(gaussian_img), plt.title("Gaussian Noise"), plt.axis('off')
plt.subplot(3, 3, 3), plt.imshow(gaussian_denoised), plt.title("Gaussian + Gaussian Filter"), plt.axis('off')

plt.subplot(3, 3, 4), plt.imshow(img), plt.title("Original"), plt.axis('off')
plt.subplot(3, 3, 5), plt.imshow(s_p_img), plt.title("Salt & Pepper Noise"), plt.axis('off')
plt.subplot(3, 3, 6), plt.imshow(s_p_denoised), plt.title("S&P + Gaussian Filter"), plt.axis('off')

plt.subplot(3, 3, 7), plt.imshow(img), plt.title("Original"), plt.axis('off')
plt.subplot(3, 3, 8), plt.imshow(speckle_img), plt.title("Speckle Noise"), plt.axis('off')
plt.subplot(3, 3, 9), plt.imshow(speckle_denoised), plt.title("Speckle + Gaussian Filter"), plt.axis('off')

plt.show()
