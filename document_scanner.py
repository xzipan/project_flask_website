import numpy as np
import cv2
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
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def document_scanner(image_path):
    image, orig, ratio = preprocess_image(image_path)
    edged = detect_edges(image)
    cnts = find_contours(edged)
    screenCnt = find_screen_contour(cnts)
    
    if screenCnt is not None:
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    else:
        print("No contour with 4 points found.")
        warped = None

    return image, orig, edged, warped, screenCnt

# def display_images(image, orig, edged, warped, screenCnt):
#     cv2.imshow("Edged", edged)
#     cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
#     cv2.imshow("Outline", image)
#     cv2.imshow("Original", imutils.resize(orig, height=500))
#     cv2.imshow("Scanned", imutils.resize(warped, height=500))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

