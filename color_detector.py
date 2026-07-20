import cv2
import numpy as np


class ColorDetector:

    def __init__(self):

        self.color_ranges = {
            "Red": [
                ((0, 120, 70), (10, 255, 255)),
                ((170, 120, 70), (180, 255, 255))
            ],
            "Green": [
                ((36, 50, 70), (89, 255, 255))
            ],
            "Blue": [
                ((90, 50, 70), (128, 255, 255))
            ],
            "Yellow": [
                ((20, 100, 100), (35, 255, 255))
            ],
            "Orange": [
                ((10, 100, 100), (20, 255, 255))
            ],
            "Purple": [
                ((129, 50, 70), (158, 255, 255))
            ],
            "White": [
                ((0, 0, 200), (180, 40, 255))
            ],
            "Black": [
                ((0, 0, 0), (180, 255, 40))
            ]
        }

    def detect(self, roi):

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        avg_bgr = np.mean(roi, axis=(0, 1)).astype(int)

        avg_rgb = (
            int(avg_bgr[2]),
            int(avg_bgr[1]),
            int(avg_bgr[0])
        )

        avg_hsv = np.mean(hsv, axis=(0, 1)).astype(int)

        detected = "Unknown"
        max_pixels = 0

        total_pixels = roi.shape[0] * roi.shape[1]

        for color, ranges in self.color_ranges.items():

            pixels = 0

            for lower, upper in ranges:

                lower = np.array(lower)
                upper = np.array(upper)

                mask = cv2.inRange(hsv, lower, upper)

                pixels += cv2.countNonZero(mask)

            if pixels > max_pixels:
                max_pixels = pixels
                detected = color

        confidence = round((max_pixels / total_pixels) * 100, 1)

        return detected, avg_rgb, tuple(avg_hsv), confidence

    def find_objects(self, frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        objects = []

        for color, ranges in self.color_ranges.items():

            for lower, upper in ranges:

                lower = np.array(lower)
                upper = np.array(upper)

                mask = cv2.inRange(hsv, lower, upper)

                contours, _ = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:

                    area = cv2.contourArea(contour)

                    if area < 500:
                        continue

                    x, y, w, h = cv2.boundingRect(contour)

                    objects.append({
                        "color": color,
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "area": area
                    })

        return objects