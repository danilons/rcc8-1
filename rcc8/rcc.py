# coding: utf-8

import numpy as np
import cv2
import nms


class RCC8(object):

    RELATIONS = {n: item for n, item in enumerate(['DC (Disconnected)',
                                                   'EC (Externally connected)',
                                                   'PO (Partially Overlapping)',
                                                   'EQ (Equal)',
                                                   'TPP (Tangential proper part)',
                                                   'TPPi (Tangential proper part inverse)',
                                                   'NTPP (Non-Tangential proper part)',
                                                   'NTPPi (Non-Tangential proper part inverse)'])}

    CLASSES = ('__background__',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    def draw_box(self, image, coords):
        (x, y, w, h) = coords
        img = np.zeros(image.shape[:2], dtype=np.uint8)
        contour = np.array([[y, x], [h, x], [h, w], [y, w]], dtype=np.float32)
        cv2.rectangle(img, (x, y), (w, h), (255), -2)
        return img, contour

    def get_relations(self, image, annotations, objects, classes=None):
        assert (len(annotations) == len(objects)), 'Annotations should have same length of objects'

        # create response objects
        classes = classes or self.CLASSES
        rcc_labels = np.zeros((1, len(self.RELATIONS)), dtype=np.uint8)
        rcc_names = []
        boxes = []

        if len(objects) < 2:
            # RCC8: EQ (Equal)
            rcc_labels[0, 3] = 1
            label = annotations[0]
            rcc_names.append((classes[label], classes[label], self.RELATIONS[3]))
            boxes.append((objects, objects))
            return rcc_labels, rcc_names, boxes

        for n1, (label1, box1) in enumerate(zip(annotations, objects)):
            for n2, (label2, box2) in enumerate(zip(annotations[n1 + 1:], objects[n1 + 1:])):
                img1, contour1 = self.draw_box(image, box1)
                img2, contour2 = self.draw_box(image, box2)

                # compute area before apply RCC8 engine
                area1, area2 = cv2.contourArea(contour1), cv2.contourArea(contour2)
                fusion = np.bitwise_or(img1, img2)

                # check overlapping
                nobjects = nms.count_objects(fusion, 255)
                if len(nobjects) == 2:
                    # RCC8: EQ (Disconnected)
                    rcc_labels[0, 0] += 1
                    rcc_names.append((classes[label1], classes[label2], self.RELATIONS[0]))
                    boxes.append((box1, box2))

                else:
                    contours, _ = cv2.findContours(fusion.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cnt_area = sum([cv2.contourArea(contour) for contour in contours])

                    (x1, y1, w1, h1) = map(int, box1)
                    (x2, y2, w2, h2) = map(int, box2)

                    if cnt_area == area1 or cnt_area == area2:
                        # RCC8: NTPP and NTPPi (Non-Tangential proper part)
                        if np.abs(x1 - x2) > 5 or np.abs(y1 - y2) > 5:
                            rcc_labels[0, 6] += 1
                            rcc_labels[0, 7] += 1

                            rcc_names.append((classes[label1], classes[label2], self.RELATIONS[6]))
                            boxes.append((box1, box2))

                            rcc_names.append((classes[label1], classes[label2], self.RELATIONS[7]))
                            boxes.append((box1, box2))

                        # RCC8: TPP and TPPi (Tangential proper part)
                        else:
                            rcc_labels[0, 4] += 1
                            rcc_labels[0, 5] += 1

                            rcc_names.append((classes[label1], classes[label2], self.RELATIONS[4]))
                            boxes.append((box1, box2))

                            rcc_names.append((classes[label1], classes[label2], self.RELATIONS[5]))
                            boxes.append((box1, box2))

                    elif np.abs(x1 - x2) > 2 or np.abs(y1 - y2) > 2:
                        # RCC8: PO (Partially Overlapping)
                        rcc_labels[0, 2] += 1
                        rcc_names.append((classes[label1], classes[label2], self.RELATIONS[2]))
                        boxes.append((box1, box2))

                    else:
                        # RCC8: EC (Externally connected)
                        rcc_labels[0, 1] += 1
                        rcc_names.append((classes[label1], classes[label2], self.RELATIONS[1]))
                        boxes.append((box1, box2))

        return rcc_labels, rcc_names, boxes
