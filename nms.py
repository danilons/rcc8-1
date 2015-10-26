# coding: utf-8
import cv2
import numpy as np


def nms(boxes, overlapThresh=0.2):
    """
    Nom max suppresion fast

    Adapted from:
    http://www.computervisionblog.com/2011/08/blazing-fast-nmsm-from-exemplar-svm.html
    """
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")


def count_objects(image, pixel_mask):
    """
    Count objects according to a pixel_mask
    """
    img = np.zeros(image.shape, dtype=np.uint8)

    if pixel_mask not in image:
        return []

    x, y = np.where(image == pixel_mask)
    img[x, y] = 255.0

    # ensure it is a binarized image
    assert len(np.unique(img)) == 2

    ftr = cv2.bilateralFilter(img, 11, 17, 17)
    blocks = []
    contours, _ = cv2.findContours(ftr.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            blocks.append(np.array([y, x, y + h, x + w]))

    return nms(np.array(blocks, dtype=np.float32))
