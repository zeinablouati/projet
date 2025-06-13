import csv
import os
from datetime import datetime

from deepface import DeepFace


def recognize_name(face_img, db_path="../dataset", model_name="VGG-Face", threshold_score=60):
    """
    Renvoie le nom reconnu pour l'image de visage (face_img),
    ou "Inconnu" si la reconnaissance échoue.
    Utilise DeepFace.find sur la base de données spécifiée.
    """
    try:
        results = DeepFace.find(
            img_path=face_img,
            db_path=db_path,
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
            score = (1 - distance/threshold) * 100
            if score >= threshold_score:
                # Nom du dossier parent de l'image trouvée
                return os.path.basename(os.path.dirname(identity_path))
    except Exception:
        pass
    return "Inconnu"


class DataLogger:
    """
    Logger simple pour enregistrer des évènements dans un CSV.
    Chaque ligne aura un timestamp et des colonnes définies par fieldnames.
    """
    def __init__(self, filename: str, fieldnames: list[str]):
        # On stocke le log à côté de ce fichier, dans ../logs/
        base = os.path.abspath(os.path.dirname(__file__))
        log_dir = os.path.normpath(os.path.join(base, "..", "logs"))
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)

        # On écrit l'en-tête (si le fichier n'existe pas encore)
        if not os.path.exists(self.path):
            with open(self.path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp"] + fieldnames)
                writer.writeheader()

        self.fieldnames = fieldnames

    def log(self, **kwargs):
        """
        Écrit une ligne de log. 
        Les clefs de kwargs doivent être dans fieldnames.
        """
        row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        for key in self.fieldnames:
            row[key] = kwargs.get(key, "")
        with open(self.path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp"] + self.fieldnames)
            writer.writerow(row)
