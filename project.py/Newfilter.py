import cv2
import numpy as np


def detect_noise_type(img_gray):
  
    if len(img_gray.shape) != 2:
        raise ValueError("detect_noise_type expects a grayscale image")

    h, w = img_gray.shape
    total_pixels = h * w
    img = img_gray

    num_zeros = np.sum(img == 0)
    num_max = np.sum(img == 255)
    sp_ratio = (num_zeros + num_max) / total_pixels

    img_f = img.astype(np.float32)
    mean = cv2.blur(img_f, (7, 7))
    mean_sq = cv2.blur(img_f * img_f, (7, 7))
    var = mean_sq - mean * mean
    var[var < 0] = 0
    std = np.sqrt(var)
    cv_map = std / (mean + 1e-3)
    cv_value = np.mean(cv_map)

    SP_THRESHOLD = 0.02    
    SPECKLE_CV_THRESHOLD = 0.5

    if sp_ratio > SP_THRESHOLD:
        noise_type = "salt_pepper"
    elif cv_value > SPECKLE_CV_THRESHOLD:
        noise_type = "speckle"
    else:
        noise_type = "gaussian"

    print(f"[INFO] sp_ratio = {sp_ratio:.4f}, cv = {cv_value:.4f}")
    print(f"[INFO] Detected noise type: {noise_type}")

    return noise_type



def restore_salt_pepper(img_gray, ksize=3, iterations=1):
    
    restored = img_gray.copy()
    for _ in range(iterations):
        restored = cv2.medianBlur(restored, ksize)
    return restored


def restore_gaussian(img_gray, use_nlm=True):
    
    if use_nlm:
        restored = cv2.fastNlMeansDenoising(
            img_gray,
            None,
            h=10,      
            templateWindowSize=7,
            searchWindowSize=21
        )
    else:
        restored = cv2.GaussianBlur(img_gray, (5, 5), 1.0)
    return restored


def restore_speckle(img_gray):
   
    img_f = img_gray.astype(np.float32)
    img_f[img_f <= 0] = 1  
    log_img = np.log(img_f)

  
    log_smooth = cv2.GaussianBlur(log_img, (5, 5), 1.0)

 
    exp_img = np.exp(log_smooth)
    exp_img = np.clip(exp_img, 0, 255).astype(np.uint8)
    return exp_img




def auto_denoise(noisy_bgr):
   
    gray = cv2.cvtColor(noisy_bgr, cv2.COLOR_BGR2GRAY)

    noise_type = detect_noise_type(gray)

    if noise_type == "salt_pepper":
        restored_gray = restore_salt_pepper(gray, ksize=3, iterations=2)

    elif noise_type == "gaussian":
        restored_gray = restore_gaussian(gray, use_nlm=True)

    elif noise_type == "speckle":
        restored_gray = restore_speckle(gray)

    else:
       
        restored_gray = gray.copy()

    return restored_gray, noise_type




if __name__ == "__main__":
    
    noisy_path = "noisy_image.png"

    noisy = cv2.imread(noisy_path)
    if noisy is None:
        raise FileNotFoundError(f"Could not open image: {noisy_path}")

    restored_gray, noise_type = auto_denoise(noisy)

    print(f"[RESULT] Final classified noise type: {noise_type}")

   
    cv2.imshow("noisy_image", noisy)
    cv2.imshow("Restored (Auto)", restored_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("restored_auto.png", restored_gray)
    print("[INFO] Restored image saved as 'restored_auto.png'")
