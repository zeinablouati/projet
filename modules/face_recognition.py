from deepface import DeepFace
import os
from config import database_path, model_name, threshold_score

def recognize_face(face_img):
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
                name = os.path.basename(os.path.dirname(identity_path))
                return name, score
    except Exception as e:
        print(f"Erreur de reconnaissance : {e}")
    return "Inconnu", 0
