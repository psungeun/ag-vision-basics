import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # -------------------------------------------------------------------------
    # 1. Image Initialization & Automated ROI Detection
    # -------------------------------------------------------------------------
    # Load the target image
    img_path = 'lettuce_test.jpg'
    img_original = cv2.imread(img_path)

    if img_original is None:
        print("[ERROR] Image files not found. Please check the file paths.")
        return
        
    img_display = img_original.copy()

    # Automated ROI generation using HSV color space (instead of manual coordinates)
    hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find the largest contour in the green mask (target plant)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_green_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_green_contour)

    # Add padding to the Bounding Box to fully enclose the plant
    pad = 40
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_original.shape[1], x + w + pad)
    y2 = min(img_original.shape[0], y + h + pad)
    rect = (x1, y1, x2 - x1, y2 - y1)

    cv2.rectangle(img_display, (x1, y1), (x2, y2), (255, 0, 0), 8)
    print(f"[INFO] Auto-ROI properly detected the target plant at: {rect}")

    # -------------------------------------------------------------------------
    # 2. Foreground Segmentation (GrabCut Algorithm)
    # -------------------------------------------------------------------------
    mask = np.zeros(img_original.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    print("[INFO] Executing GrabCut algorithm to remove soil and background noise...")
    
    # Apply Bounding Box-based GrabCut
    cv2.grabCut(img_original, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Retain only definite (1) and probable (3) foreground pixels
    binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_segmented = img_original * binary_mask[:, :, np.newaxis]
    
    print("[INFO] Target plant successfully segmented from the background.")

    # -------------------------------------------------------------------------
    # 3. Phenotypic Trait Extraction (Contour & Area)
    # -------------------------------------------------------------------------
    gray_segmented = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_segmented, 1, 255, cv2.THRESH_BINARY)
    final_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not final_contours:
        print("[WARNING] No contours detected in the segmented area.")
        return

    # Extract the final area from the segmented result
    target_contour = max(final_contours, key=cv2.contourArea)
    leaf_area = cv2.contourArea(target_contour)
    
    print(f"[INFO] Phenotypic Trait Extracted -> Plant Area: {leaf_area} pixels.")

    # -------------------------------------------------------------------------
    # 4. Visualization
    # -------------------------------------------------------------------------
    print("[INFO] Generating visualization for the extracted phenotypic traits...")

    # Draw the contour and text for the area
    cv2.drawContours(img_segmented, [target_contour], -1, (0, 255, 0), 8)
    text = f"Plant Area: {leaf_area:.1f} px"
    cv2.putText(img_segmented, text, (x1, max(y1 - 30, 50)), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 6)

    # Plot the results side-by-side
    plt.figure(figsize=(24, 8))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    plt.title("Step 1. Auto-ROI Detection (HSV)", fontsize=18)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(binary_mask, cmap='gray')
    plt.title("Step 2. GrabCut Foreground Mask", fontsize=18)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img_segmented, cv2.COLOR_BGR2RGB))
    plt.title("Step 3. Extracted Phenotypic Trait (Area)", fontsize=18)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
