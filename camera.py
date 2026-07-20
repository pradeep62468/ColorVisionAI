import cv2


class Camera:

    def __init__(self, camera_index=0):
        """
        Initialize the webcam.
        camera_index = 0 refers to the default webcam.
        """
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise Exception("Unable to access the camera.")

    def get_frame(self):
        """
        Capture a single frame.
        Returns:
            ret (bool)
            frame (numpy array)
        """
        return self.cap.read()

    def release(self):
        """
        Release the camera resource.
        """
        if self.cap.isOpened():
            self.cap.release()