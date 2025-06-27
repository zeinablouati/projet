import csv
import os
from deepface import DeepFace

class EmotionLogger:
    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = fieldnames
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, **kwargs):
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(kwargs)


def recognize_name(face_img, db_path, model_name="VGG-Face", threshold_score=60):
    try:
        result = DeepFace.find(img_path=face_img, db_path=db_path, model_name=model_name, enforce_detection=False)
        if result and isinstance(result, list) and len(result) > 0 and not result[0].empty:
            name = os.path.basename(result[0].iloc[0]['identity']).split('/')[0]
            return name
        else:
            return "Inconnu"
    except Exception:
        return "Inconnu"
