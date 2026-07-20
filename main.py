from color_detector import ColorDetector
from ui import draw_header
from config import *

import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

detector = ColorDetector()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the camera like a mirror
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    height, width = frame.shape[:2]

    # Define the size of the target box
    box_size = 120

    # Calculate the center position
    x1 = width // 2 - box_size // 2
    y1 = height // 2 - box_size // 2

    x2 = x1 + box_size
    y2 = y1 + box_size

    # Draw the target box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Extract ROI
    roi = frame[y1:y2, x1:x2]

    # Detect color
    detected_color, rgb, hsv, confidence = detector.detect(roi)

    # Detect all colored objects in the frame
    objects = detector.find_objects(frame)

    # Convert RGB to OpenCV's BGR format
    preview_color = (rgb[2], rgb[1], rgb[0])

    # Display information
    cv2.putText(
        frame,
        f"Color : {detected_color}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"RGB : {rgb}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"HSV : {hsv}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Confidence : {confidence}%",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    # Draw filled color preview box
    cv2.rectangle(
        frame,
        (width - 180, 20),
        (width - 30, 100),
        preview_color,
        -1
    )

    # Draw border
    cv2.rectangle(
        frame,
        (width - 180, 20),
        (width - 30, 100),
        (255, 255, 255),
        2
    )

    # Label
    cv2.putText(
        frame,
        "Preview",
        (width - 170, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    draw_header(frame, width)

    # Draw bounding boxes around detected objects
    for obj in objects:

        x = obj["x"]
        y = obj["y"]
        w = obj["w"]
        h = obj["h"]
        color = obj["color"]

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        color,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )
    cv2.imshow("Color Vision AI", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()