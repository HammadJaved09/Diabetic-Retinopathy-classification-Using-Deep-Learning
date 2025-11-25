import cv2
import numpy as np

IMG_SIZE = 300  

def crop_image_from_gray(img, tol=7):
    

    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(axis=1), mask.any(axis=0))]

    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        if mask.any():
            img1 = img[:, :, 0][np.ix_(mask.any(axis=1), mask.any(axis=0))]
            img2 = img[:, :, 1][np.ix_(mask.any(axis=1), mask.any(axis=0))]
            img3 = img[:, :, 2][np.ix_(mask.any(axis=1), mask.any(axis=0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img
    else:
        return img  # Unrecognized format, return as-is


def preprocess(image_path):
    """
    Reads an image from path, crops and enhances it, resizes it, and returns the result.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from path: {image_path}")
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)
    return image
