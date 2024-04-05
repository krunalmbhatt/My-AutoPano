#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

import numpy as np
import cv2
import os

# Add any python libraries here


def anms(corners, Nbest):
    strong_corners = np.argwhere(corners > 0)
    N_strong = len(strong_corners)
    print(f"Number of strong corners before ANMS: {N_strong}")  # Debug print
    thresholded_corners = corners * (corners > 0.15 * corners.max())  # Apply a threshold
    strong_corners = np.argwhere(thresholded_corners)
    N_strong = len(strong_corners)
    print(f"Number of strong corners after supression: {N_strong}")  # Debug print
    
    r = np.inf * np.ones(N_strong)

    for i in range(N_strong):
        for j in range(N_strong):
            if corners[strong_corners[i][0], strong_corners[i][1]] > corners[strong_corners[j][0], strong_corners[j][1]]:
                ED = np.sum((strong_corners[i] - strong_corners[j]) ** 2)
                if ED < r[i]:
                    r[i] = ED
    
    sorted_indices = np.argsort(-r)  # Sort in descending order
    best_corners_indices = sorted_indices[:Nbest]
    best_corners = strong_corners[best_corners_indices]
    print("ANMS completed.")
    return best_corners

def convert_to_keypoint_objects(keypoints):
    # The size parameter is set to 10, which is arbitrary.
    # You may adjust the size for better visualization if needed.
    return [cv2.KeyPoint(x=float(point[1]), y=float(point[0]), size=10) for point in keypoints]

def create_feature_descriptors(img, keypoints, patch_size=41, final_size=8):
    descriptors = []
    half_patch = patch_size // 2

    for x, y in keypoints:
        # Check if the patch around the keypoint is within the image boundaries
        if y - half_patch >= 0 and y + half_patch < img.shape[0] and x - half_patch >= 0 and x + half_patch < img.shape[1]:
            # Extract the patch centered around the keypoint
            patch = img[y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1]

            # Apply Gaussian blur
            blurred_patch = cv2.GaussianBlur(patch, (5,5), 0)

            # Subsample the blurred patch to reduce dimensionality
            subsampled_patch = cv2.resize(blurred_patch, (final_size, final_size), interpolation=cv2.INTER_AREA)

            # Flatten and standardize the patch to form the descriptor
            descriptor = subsampled_patch.flatten()
            descriptor = (descriptor - descriptor.mean()) / (descriptor.std() + 1e-5)
            descriptors.append(descriptor)

    return np.array(descriptors)


def create_feature_descriptors(img, keypoints, patch_size=41, final_size=8):
    descriptors = []
    half_patch = patch_size // 2
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])  # KeyPoint coordinates
        if y - half_patch >= 0 and y + half_patch < img.shape[0] and x - half_patch >= 0 and x + half_patch < img.shape[1]:
            patch = img[y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1]
            blurred_patch = cv2.GaussianBlur(patch, (5, 5), 0)
            subsampled_patch = cv2.resize(blurred_patch, (final_size, final_size), interpolation=cv2.INTER_AREA)
            descriptor = subsampled_patch.flatten()
            mean = descriptor.mean()
            std = descriptor.std() if descriptor.std() != 0 else 1
            descriptor = (descriptor - mean) / std
            descriptors.append(descriptor)
    return np.array(descriptors)



def visualize_patches(img, keypoints, patch_size=41):
    half_patch = patch_size // 2
    vis_img = np.copy(img)

    for y, x in keypoints:  # Notice that we're using y, x here because np.argwhere returns coordinates in (row, column) format
        top_left = (int(x) - half_patch, int(y) - half_patch)  # Convert to (x, y) for drawing
        bottom_right = (int(x) + half_patch, int(y) + half_patch)
        cv2.rectangle(vis_img, top_left, bottom_right, (0, 255, 0), 1)

    return vis_img



def match_features(descriptors1, descriptors2, ratio_threshold=0.8):
    matches = []
    for i, descriptor1 in enumerate(descriptors1):
        # Compute SSD between descriptor1 and all descriptors in image 2
        distances = np.sum((descriptors2 - descriptor1) ** 2, axis=1)
        
        # Sort the distances
        sorted_indices = np.argsort(distances)
        best_match_index = sorted_indices[0]
        second_best_match_index = sorted_indices[1]

        # Compute the ratio of the best and second-best match
        ratio = distances[best_match_index] / distances[second_best_match_index]
        
        # If the ratio is below the threshold, accept the match
        if ratio < ratio_threshold:
            matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=best_match_index, _distance=distances[best_match_index]))
    return matches

def match_features(descriptors1, descriptors2, ratio_threshold=0.7):
    # Convert descriptors to type float32 if not already
    descriptors1 = np.float32(descriptors1)
    descriptors2 = np.float32(descriptors2)

    # Initialize the BFMatcher with default parameters
    bf = cv2.BFMatcher()

    # Check if descriptors are empty
    if descriptors1.shape[0] == 0 or descriptors2.shape[0] == 0:
        return []

    # Find the two nearest neighbors for each descriptor
    knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply the ratio test to filter matches
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    return good_matches


def match_features_flann(descriptors1, descriptors2, ratio_threshold=0.75):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    descriptors1 = np.float32(descriptors1)
    descriptors2 = np.float32(descriptors2)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    return good_matches




# Function to visualize feature correspondences
def draw_feature_matches(img1, keypoints1, img2, keypoints2, matches):
    matched_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
    cv2.imshow('Feature Correspondences', matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return matched_image
    #cv2.imwrite('feature_correspondences.png', matched_image)

def keypoints_to_array(keypoints):
    return np.array([kp.pt for kp in keypoints])

def keypoints_to_np_array(keypoints):
    """Convert a list of cv2.KeyPoint objects to a NumPy array."""
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return points.reshape(-1, 1, 2)


def ransac(keypoints1, keypoints2, matches, iterations=1000, threshold=5):
    # Convert keypoints to coordinates
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    max_inliers = []
    best_H = None

    for _ in range(iterations):
        # Randomly select 4 matches
        selected_matches = np.random.choice(matches, 4, replace=False)
        selected_points1 = np.float32([keypoints1[m.queryIdx].pt for m in selected_matches]).reshape(-1, 1, 2)
        selected_points2 = np.float32([keypoints2[m.trainIdx].pt for m in selected_matches]).reshape(-1, 1, 2)

        # Compute homography
        #H, status = cv2.findHomography(selected_points1, selected_points2, method=20)
        H, status = cv2.findHomography(selected_points1, selected_points2, cv2.RANSAC, threshold)

        
        
        if H is None:
            continue

        # Project points1 to image2 plane
        transformed_points = cv2.perspectiveTransform(points1, H)

        # Compute inliers
        distances = np.sqrt(np.sum((points2 - transformed_points) ** 2, axis=2))
        inliers = distances < threshold

        # Update best homography
        if np.sum(inliers) > len(max_inliers):
            max_inliers = inliers
            best_H = H

    # Refine homography with all inliers
    inlier_matches = [matches[i] for i in range(len(matches)) if max_inliers[i]]
    if len(inlier_matches) > 4:
        final_points1 = np.float32([keypoints1[m.queryIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)
        final_points2 = np.float32([keypoints2[m.trainIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)
        #best_H, _ = cv2.findHomography(final_points1, final_points2, method=0)
        best_H, _ = cv2.findHomography(final_points1, final_points2, cv2.RANSAC, threshold)


    return best_H, inlier_matches

def warpTwoImages(img1, img2, H):
    """ Warp img2 to img1 with the homography matrix H """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # Translation matrix
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax - xmin, ymax - ymin))
    result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1
    return result


def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    # """
    # Corner Detection
    # Save Corner detection output as corners.png
    # """
    # print('Corner Detection ...')
    # corners = []
    # for i, img in enumerate(imgs):
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     gray = np.float32(gray)
    #     corner = cv2.cornerHarris(gray, 2, 3, 0.04)
    #     corners.append(corner)
    #     corner = cv2.dilate(corner, None)
    #     img[corner > 0.01 * corner.max()] = [0, 0, 255]
    #     output_file_name = f"set1_out_{i}.jpg"
    #     cv2.imwrite(output_file_name, img) 
    # print('Done!')
    
    # # Display the first image with corners
    # cv2.imshow('Corners on first image', imgs[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()  # Close the window after a key press

    # print('Displayed corners on the first image.')
    
    """
    Read a set of images for Panorama stitching
    """
    
    # # Read a set of images for Panorama stitching
    # print('Reading images ...')
    # #img_path = '../phase1/Data/Train/Set1/'
    # img_path = '../phase1/Data/Train/CustomSet2/'
    # #/home/jesulona/RBE549/RBE-549-Classical-and-Deep-Learning-Approaches-for-Geometric-Computer-Vision/Project1MyAutoPano/phase1/Data/Train/CustomSet1
    # files = sorted(os.listdir(img_path))  # Make sure files are sorted in numeric order
    # imgs = [cv2.imread(img_path + file) for file in files if file.endswith('.jpg')]
    
    # print(f'Number of images: {len(imgs)}')
    # cv2.imshow('img', imgs[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 
    # print('Done!')
    
    # Read a set of images for Panorama stitching
    print('Reading images ...')
    #img_path = '../phase1/Data/Train/Set1/'
    img_path = '../phase1/Data/Train/CustomSet1/'  # Update as needed
    files = sorted(os.listdir(img_path))  # Ensure sorted file list
    print("Detected files:", files)  # Debug: Print detected files

    # Load and resize images
    imgs = []
    target_width = 800  # or any width you prefer
    for file in files:
        if file.endswith('.jpg'):
            img = cv2.imread(img_path + file)
            if img.shape[:2][1] >1000:
                height, width = img.shape[:2]
                aspect_ratio = width / height
                target_height = int(target_width / aspect_ratio)
                resized_img = cv2.resize(img, (target_width, target_height))
                imgs.append(resized_img)
            else:
                imgs.append(img)

    # Check if any images are loaded and display
    if len(imgs) == 0:
        print("No images found. Check the path and file extensions.")
    else:
        print(f'Number of images: {len(imgs)}')
        cv2.imshow('img', imgs[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print('Done!')

    """
    Corner Detection
    Save Corner detection output as corners.png 
    """
    
    # Perform corner detection on all images
    print('Corner Detection ...')
    corners = []  # Store corner data for all images
    for i, img in enumerate(imgs):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        #dst = cv2.dilate(dst,None)
        corners.append(dst)
    
    # Make copies of the original images for drawing
    imgs_with_corners = [np.copy(img) for img in imgs]
    for i, corner in enumerate(corners):
        imgs_with_corners[i][corner > 0.01 * corner.max()] = [0, 0, 255]
        output_file_name = f"set1_out_{i+1}.jpg"
        cv2.imwrite(output_file_name, imgs_with_corners[i])  # Save the image with corners marked

    # Display the first image with corners
    cv2.imshow('Corners on first image', imgs_with_corners[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('Displayed corners on the first image.')
    
    
    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """
    corner_images = corners 
    imgs_with_anms = [np.copy(img) for img in imgs]  # Clean copies for ANMS results
    Nbest = 100 # Define the number of best corners needed
    print("Starting ANMS on all images...")

    # Apply ANMS to each image and store the best corners
    all_best_corners = [anms(corner_img, Nbest) for corner_img in corner_images]
    
    # Draw the best corners on the clean copy of each image
    for img_index, best_corners in enumerate(all_best_corners):
        for y, x in best_corners:
            cv2.circle(imgs_with_anms[img_index], (x, y), radius=3, color=(0, 0, 255), thickness=1)  # Red dot

    # Display the first image with ANMS corners
    cv2.imshow('ANMS Corners', imgs_with_anms[0])
    cv2.waitKey(0)  # Wait for any key press
    cv2.destroyAllWindows()  # Destroy the window after key press to avoid hanging
    print('ANMS applied and displayed on the first image.')

    # Save the first image with ANMS corners 
    cv2.imwrite('anms_first_image.png', imgs_with_anms[0])
    
    

    #If you want to save all images with ANMS corners, uncomment the following lines:
    for i, img in enumerate(imgs_with_anms):
        cv2.imwrite(f'anms_image_{i+1}.png', img)

    """
        Feature Descriptors
        Save Feature Descriptor output as FD.png
    """
    all_feature_descriptors = []  # To store feature descriptors for all images
    for img_index, img in enumerate(imgs):
        # keypoints = all_best_corners[img_index]
        # feature_descriptors = create_feature_descriptors(img, keypoints)
        # all_feature_descriptors.append(feature_descriptors)
        keypoints_array = all_best_corners[img_index]
        # Convert numpy array to cv2.KeyPoint objects
        keypoints = [cv2.KeyPoint(x=float(point[1]), y=float(point[0]), size=10) for point in keypoints_array]
        feature_descriptors = create_feature_descriptors(img, keypoints)
        all_feature_descriptors.append(feature_descriptors)


    # Visualize patches for the first image
    patches_img = visualize_patches(imgs_with_anms[0], all_best_corners[0], patch_size=41)
    cv2.imshow('Feature Patches', patches_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the visualization
    cv2.imwrite('feature_patches.png', patches_img)

    """
    Feature Matching
    Save Feature Matching output as matching.png
    """    
    # Convert keypoints to cv2.KeyPoint objects for drawing matches
    keypoints_img1 = convert_to_keypoint_objects(all_best_corners[0])
    keypoints_img2 = convert_to_keypoint_objects(all_best_corners[1])

    # Now match features and draw feature correspondences
    descriptors_img1 = all_feature_descriptors[0]
    descriptors_img2 = all_feature_descriptors[1]
    
    # Match features
    matches = match_features(descriptors_img1, descriptors_img2)

    # Draw feature correspondences
    fm = draw_feature_matches(imgs[0], keypoints_img1, imgs[1], keypoints_img2, matches)
    
    cv2.imwrite('feature_correspondences.png', fm )




    """
    Refine: RANSAC, Estimate Homography
    """
    iterations = 5000

    # Convert keypoints to NumPy array of coordinates for RANSAC
    # coords_img1 = keypoints_to_np_array(keypoints_img1)
    # coords_img2 = keypoints_to_np_array(keypoints_img2)
    
    coords_img1 = keypoints_img1
    coords_img2 = keypoints_img2
    
    # Call RANSAC
    h, inlier_matches = ransac(keypoints_img1, keypoints_img2, matches, iterations, threshold=5)
    #h, inlienmatches = ransac(keypoints_img1, keypoints_img2, iterations, threshold=5)

    
    #print(f"Homography matrix:\n{h}")


    if h is not None:
        # Draw feature correspondences for inliers
        Ransac_Output = draw_feature_matches(imgs[0], keypoints_img1, imgs[1], keypoints_img2, inlier_matches)
        cv2.imwrite('Ransac_Output.png', Ransac_Output)
    else:
        print("RANSAC did not find a suitable homography.")



    """
    Image Warping + Blending
    Save Panorama output as mypano.png
    """
    # Assuming h is the homography matrix from image1 to image2
    panorama = warpTwoImages(imgs[0], imgs[1], h)

    #display result
    cv2.imshow('panorama', panorama )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the panorama
    cv2.imwrite('mypano.png', panorama)

    

if __name__ == "__main__":
    main()
