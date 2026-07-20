import cv2
from config import *

def draw_header(frame, width):

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 50),
        GRAY,
        -1
    )

    cv2.putText(
        frame,
        "ColorVision AI",
        (20, 32),
        FONT,
        0.9,
        YELLOW,
        2
    )

    cv2.putText(
        frame,
        "Camera : LIVE",
        (width - 180, 32),
        FONT,
        0.6,
        GREEN,
        2
    )