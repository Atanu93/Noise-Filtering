import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("original_image.png")  # change file name if needed
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB

def add_gaussian_noise(image, mean=0, var=20):
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + gaussian
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def add_salt_pepper_noise(image, prob=0.02):
    noisy = np.copy(image)
    rnd = np.random.rand(image.shape[0], image.shape[1])
    noisy[rnd < prob] = 0       # pepper
    noisy[rnd > 1 - prob] = 255 # salt
    return noisy

def add_speckle_noise(image):
    noise = np.random.randn(*image.shape)
    noisy = image + image * noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

gaussian_img = add_gaussian_noise(img)
s_p_img = add_salt_pepper_noise(img)
speckle_img = add_speckle_noise(img)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1), plt.imshow(img), plt.title("Original")
plt.subplot(2, 2, 2), plt.imshow(gaussian_img), plt.title("Gaussian Noise")
plt.subplot(2, 2, 3), plt.imshow(s_p_img), plt.title("Salt & Pepper Noise")
plt.subplot(2, 2, 4), plt.imshow(speckle_img), plt.title("Speckle Noise")
plt.show(block=True)

cv2.imwrite("gaussian_noise.jpg", cv2.cvtColor(gaussian_img, cv2.COLOR_RGB2BGR))
cv2.imwrite("salt_pepper_noise.jpg", cv2.cvtColor(s_p_img, cv2.COLOR_RGB2BGR))
cv2.imwrite("speckle_noise.jpg", cv2.cvtColor(speckle_img, cv2.COLOR_RGB2BGR))

print(" Noisy images saved as gaussian_noise.jpg, salt_pepper_noise.jpg, speckle_noise.jpg")

