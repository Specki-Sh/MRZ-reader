import numpy as np
import cv2

color_white = (255, 255, 255)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 27))


def smooth_image(gray):
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)


# https://pythongeeks.org/sobel-and-scharr-operator-in-opencv/
def compute_gradient(black_hat):
    gradX = cv2.Sobel(black_hat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    return (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")


# Размываем текст чтобы слепить его в единный камок
def apply_closing_operations(gradX, rectKernel, sqKernel):
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(
        gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=4)
    return thresh


# При размытие мы могли соъединить текст с границами, убераем 5% слева и справа
def remove_border_pixels(thresh, image):
    p = int(image.shape[1] * 0.05)
    thresh[:, 0:p] = 0
    thresh[:, image.shape[1] - p:] = 0
    return thresh


def find_contours(thresh):
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
    return sorted(contours, key=cv2.contourArea, reverse=True)


def resize_image(img, height):
    width = int(img.shape[1] * height / img.shape[0])
    return cv2.resize(img, (width, height))


def get_mrz_image(img):
    image = resize_image(img, 600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_hat = smooth_image(gray)
    gradX = compute_gradient(black_hat)
    thresh = apply_closing_operations(gradX, rectKernel, sqKernel)
    thresh = remove_border_pixels(thresh, image)
    contours = find_contours(thresh)
    return extract_roi(contours, gray, image)


def extract_roi(contours, gray, image):
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        crWidth = w / float(gray.shape[1])

        if ar > 5 and crWidth > 0.75:
            pX = int((x + w) * 0.03)
            pY = int((y + h) * 0.03)
            (x, y) = (x - pX, y - pY)
            (w, h) = (w + (pX * 2), h + (pY * 2))

            roi = image[y:y + h, x:x + w].copy()
            break

    return roi


def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (
        width / 2,
        height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    rotated_mat = cv2.warpAffine(mat, rotation_mat,
                                 (bound_w, bound_h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=color_white,
                                 )
    return rotated_mat


def convert_to_binary(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 255)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    bg = cv2.morphologyEx(img, cv2.MORPH_DILATE, se)
    out_gray = cv2.divide(img, bg, scale=255)
    out_binary = cv2.threshold(out_gray, 55, 255, cv2.THRESH_OTSU)[1]
    return out_binary
