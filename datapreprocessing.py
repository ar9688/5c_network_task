import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split


def preprocess_images(image_dir, mask_dir):
    images = []
    masks = []
#implementinng clache

    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(image_dir, filename), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(os.path.join(mask_dir, filename), cv2.IMREAD_GRAYSCALE)

            # CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_clahe = clahe.apply(img)
            images.append(img_clahe)
            masks.append(mask)

    return np.array(images), np.array(masks)


images, masks = preprocess_images('path_to_images', 'path_to_masks')
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)
