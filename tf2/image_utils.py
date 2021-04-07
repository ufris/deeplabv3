import numpy as np
import cv2, os, skimage
import matplotlib.pyplot as plt

# for multi preprocess test
def preprocess_image(x, mode):
    x = x.astype(np.float32)
    if mode == 'tf':
        x /= 127.5
        x -= 1.
        return x
    elif mode == 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x

def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(image.shape) == 3:
        img2 = clahe.apply(image[:, :, 0])
    else:
        img2 = clahe.apply(image[:, :])
    image = np.stack([img2, img2, img2], axis=2)
    return image

def normalize(image):
    image = cv2.normalize(image,image,0,255,cv2.NORM_MINMAX)
    return image


def histogram_match(image,source_img):
    def hist_match(source, template):
        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        return interp_t_values[bin_idx].reshape(oldshape)

    source = cv2.imread(source_img)

    matched = hist_match(image, source)
    matched = np.cast[np.uint8](matched)

    return matched

def sharpen(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(img.shape) == 3:
        img = clahe.apply(img[:, :, 0])
    else:
        img = clahe.apply(img[:, :])

    sharpen = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")

    sharpen_img = cv2.filter2D(img, -1, sharpen)
    img = np.stack([sharpen_img,sharpen_img,sharpen_img], axis=2)

    return img

def gaussian_noise(img):
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

   if len(img.shape) == 3:
       img2 = clahe.apply(img[:, :, 0])
   else:
       img2 = clahe.apply(img[:, :])

   img2 = np.stack([img2, img2, img2], axis=2)
   img2 = skimage.util.random_noise(img2, mode="gaussian")
   img2 = (img2 * 255).astype(np.uint8)

   return img2
