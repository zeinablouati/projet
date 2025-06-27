import cv2
import numpy as np
import time

class PeopleCounter:
    def __init__(self):
        self.count = 0
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.entrance_line = 0.3
        self.last_detection_time = time.time()
        self.cooldown = 1.0  # secondes

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = self.bg_subtractor.apply(gray)
        kernel = np.ones((5,5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = frame.shape[:2]
        line_y = int(h * self.entrance_line)
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)

        current_time = time.time()

        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            center_y = y + h // 2
            if center_y < line_y and current_time - self.last_detection_time > self.cooldown:
                self.count += 1
                self.last_detection_time = current_time

        cv2.putText(frame, f"Count: {self.count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame
