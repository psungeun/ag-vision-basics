import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust_gamma(image, gamma=1.0):
    # -------------------------------------------------------------------------
    # Helper Function: Non-linear Gamma Correction
    # -------------------------------------------------------------------------
    inv_gamma = 1.0 / gamma
    normalized = image / 255.0
    corrected = np.uint8(255 * (normalized ** inv_gamma))
    return corrected

def main():
    # -------------------------------------------------------------------------
    # 1. Image Initialization
    # -------------------------------------------------------------------------
    img_path = 'greenhouse_crop.jpg'
    img_original = cv2.imread(img_path)

    if img_original is None:
        print(f"[ERROR] Image '{img_path}' not found. Please check the path.")
        return
        
    print("[INFO] Successfully loaded the original crop image.")

    # -------------------------------------------------------------------------
    # 2. Noise Reduction (Gaussian Blur)
    # -------------------------------------------------------------------------
    print("[INFO] Applying Gaussian Blur (5x5) to reduce high-frequency noise...")
    img_blurred = cv2.GaussianBlur(img_original, (5, 5), 0.0)

    # -------------------------------------------------------------------------
    # 3. Illumination Normalization (Gamma Correction)
    # -------------------------------------------------------------------------
    # [수정됨] Gamma < 1.0 : 어둡게 만들어 과노출(Highlight) 복원
    gamma_value_darken = 0.5
    print(f"[INFO] Applying Gamma Correction (Gamma={gamma_value_darken}) to recover highlight details...")
    img_gamma_dark = adjust_gamma(img_blurred, gamma=gamma_value_darken)

    # [수정됨] Gamma > 1.0 : 밝게 만들어 그림자(Shadow) 복원
    gamma_value_brighten = 2.0
    print(f"[INFO] Applying Gamma Correction (Gamma={gamma_value_brighten}) to recover shadow details...")
    img_gamma_bright = adjust_gamma(img_blurred, gamma=gamma_value_brighten)

    # -------------------------------------------------------------------------
    # 4. Visualization
    # -------------------------------------------------------------------------
    print("[INFO] Generating visualization for preprocessing steps...")

    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    plt.title("1. Original Image\n(Irregular Lighting)", fontsize=14)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(img_blurred, cv2.COLOR_BGR2RGB))
    plt.title("2. Gaussian Blur\n(Noise Reduction)", fontsize=14)
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(img_gamma_dark, cv2.COLOR_BGR2RGB))
    plt.title(f"3. Gamma Corrected\n(Darken: Gamma={gamma_value_darken})", fontsize=14)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(img_gamma_bright, cv2.COLOR_BGR2RGB))
    plt.title(f"4. Gamma Corrected\n(Brighten: Gamma={gamma_value_brighten})", fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
