import cv2
import os
import time
from deepface import DeepFace
#from serial import Serial

#esp32 = Serial("COM6", 9600)
#time.sleep(2)

def recognize_faces_realtime(database_path="../dataset", model_name="VGG-Face", threshold_score=60):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    tracked_faces = []  # [ {x, y, w, h, name, last_seen} ]
    cooldown_time = 5  # secondes

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        current_time = time.time()

        for (x, y, w, h) in faces:
            center = (x + w // 2, y + h // 2)
            matched = False

            # Essayer d'associer avec un visage déjà reconnu
            for face in tracked_faces:
                fx, fy, fw, fh = face["coords"]
                f_center = (fx + fw // 2, fy + fh // 2)

                if abs(center[0] - f_center[0]) < 50 and abs(center[1] - f_center[1]) < 50:
                    matched = True
                    if current_time - face["last_seen"] < cooldown_time:
                        name = face["name"]
                    else:
                        face_img = frame[y:y+h, x:x+w]
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
                                else:
                                    name = "Inconnu"
                            else:
                                name = "Inconnu"
                        except Exception:
                            name = "Erreur"

                        face["name"] = name
                        face["last_seen"] = current_time

                    face["coords"] = (x, y, w, h)
                    break

            # Si pas de correspondance, on ajoute un nouveau visage
            if not matched:
                try:
                    face_img = frame[y:y+h, x:x+w]
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
                        else:
                            name = "Inconnu"
                    else:
                        name = "Inconnu"
                except Exception:
                    name = "Erreur"

                tracked_faces.append({
                    "coords": (x, y, w, h),
                    "name": name,
                    "last_seen": current_time
                })

            # Affichage
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Reconnaissance en temps réel [Q pour quitter]", frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces_realtime()
