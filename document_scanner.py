import cv2
import numpy as np
import imutils

def preprocess_image(image_path, height=500):
    image = cv2.imread(image_path)
    ratio = image.shape[0] / float(height)
    orig = image.copy()
    image = imutils.resize(image, height=height)

    return image, orig, ratio

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return cv2.Canny(gray, 75, 200)

def find_contours(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

def find_screen_contour(contours):
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    dst = np.array([
        [0, 0],
        [1356, 0],
        [1356, 1919],
        [0, 1919]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (1357, 1920))

    return warped

def document_scanner(image_path):
    image, orig, ratio = preprocess_image(image_path)
    edged = detect_edges(image)
    cnts = find_contours(edged)
    screenCnt = find_screen_contour(cnts)
    if screenCnt is not None:
        print("Document warped.")
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
        return warped

    else:
        print("No contour with 4 points found.")
        warped = None
        return warped