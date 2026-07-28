import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # -------------------------------------------------------------------------
    # 1. Image Initialization & Cropping (ROI Extraction)
    # -------------------------------------------------------------------------
    # Load the model image and scene image
    # Recommended: '8R 410 Tractor_2.jpeg' for model, '8R410_test_2.jpeg' for scene
    img_model_full = cv2.imread('8R410Tractor.jpeg')
    img_scene = cv2.imread('8R410_test.jpeg')

    if img_model_full is None or img_scene is None:
        print("[ERROR] Image files not found. Please check the file paths.")
        return

    # Extract Region of Interest (ROI) to remove background noise
    # Adjust these coordinates [y1:y2, x1:x2] based on the actual tractor position in the image
    img_model_cropped = img_model_full[200:800, 150:900] 

    # Convert color space from BGR to Grayscale for feature extraction
    gray_model = cv2.cvtColor(img_model_cropped, cv2.COLOR_BGR2GRAY)
    gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------------------------------
    # 2. Keypoint Detection and Feature Description (SIFT)
    # -------------------------------------------------------------------------
    sift = cv2.SIFT_create()

    kp_model, des_model = sift.detectAndCompute(gray_model, None)
    kp_scene, des_scene = sift.detectAndCompute(gray_scene, None)

    print(f"[INFO] Extracted {len(kp_model)} features from Cropped Model Image.")
    print(f"[INFO] Extracted {len(kp_scene)} features from Scene Image.")

    # -------------------------------------------------------------------------
    # 3. Approximate Nearest Neighbor Matching (FLANN)
    # -------------------------------------------------------------------------
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
    knn_matches = flann_matcher.knnMatch(des_model, des_scene, k=2)

    # -------------------------------------------------------------------------
    # 4. Outlier Rejection Step I: Lowe's Ratio Test
    # -------------------------------------------------------------------------
    ratio_threshold = 0.7
    putative_matches = []

    for best_match, second_best_match in knn_matches:
        if best_match.distance < ratio_threshold * second_best_match.distance:
            putative_matches.append(best_match)

    print(f"[INFO] Retained {len(putative_matches)} putative matches post-Ratio Test.")

    if len(putative_matches) < 4:
        print("[WARNING] Not enough matches to compute Homography.")
        return

    # -------------------------------------------------------------------------
    # 5. Outlier Rejection Step II: RANSAC Algorithm
    # -------------------------------------------------------------------------
    pts_model = np.float32([kp_model[m.queryIdx].pt for m in putative_matches]).reshape(-1, 1, 2)
    pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in putative_matches]).reshape(-1, 1, 2)

    # Compute Homography to extract the inlier mask
    H_matrix, inlier_mask = cv2.findHomography(pts_model, pts_scene, cv2.RANSAC, 5.0)
    matches_mask = inlier_mask.ravel().tolist()

    robust_inliers = [putative_matches[i] for i in range(len(putative_matches)) if matches_mask[i] == 1]
    
    print(f"[INFO] Robust correspondences identified via RANSAC: {len(robust_inliers)} inliers.")

    # -------------------------------------------------------------------------
    # 6. Visualization
    # -------------------------------------------------------------------------
    print("[INFO] Generating visualization for the robust spatial correspondences...")

    # Draw all robust RANSAC inliers
    img_matches = cv2.drawMatches(
        img_model_cropped, kp_model, 
        img_scene, kp_scene, 
        robust_inliers, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0)
    )

    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f"SIFT + RANSAC Tractor Feature Matching ({len(robust_inliers)} Inliers)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
