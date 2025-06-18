from deepface import DeepFace
import cv2, time
import os
from config import database_path, model_name, threshold_score

class FaceAnalyzer:
    def __init__(self, max_faces=3):
        self.max_faces = max_faces

    def recognize_face(self, face_img):
        try:
            results = DeepFace.find(
                img_path=face_img,
                db_path=database_path,
                model_name=model_name,
                enforce_detection=False
            )
            df = results[0]
            if not df.empty:
                df = df.sort_values(by="distance", ascending=True).reset_index(drop=True)
                best = df.iloc[0]
                identity_path = best["identity"]
                distance = best["distance"]
                threshold = best.get("threshold", 0.4)
                score = (1 - distance / threshold) * 100
                if score >= threshold_score:
                    return os.path.basename(os.path.dirname(identity_path))
        except Exception as e:
            print("Erreur de reconnaissance :", e)
        return "Inconnu"

    def analyze(self, frame):
        try:
            results = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False,
            )
            if not isinstance(results, list):
                results = [results]
            
            for res in results:
                # Extraire visage à partir des coordonnées
                region = res.get("region", {})
                if all(k in region for k in ("x", "y", "w", "h")):
                    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                    face_crop = frame[y:y+h, x:x+w]
                    name = self.recognize_face(face_crop)
                    res["name"] = name
                else:
                    res["name"] = "Inconnu"

            return results[:self.max_faces]

        except Exception as e:
            print("Erreur d'analyse :", e)
            return []