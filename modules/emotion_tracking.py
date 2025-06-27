import cv2
import numpy as np
from datetime import datetime
from collections import deque
from deepface import DeepFace
from modules.face_recognition import recognize_face  # Import de votre fonction existante

# === Configuration ===
DATABASE_PATH = "../dataset"
MODEL_NAME = "VGG-Face"  # Ou "Facenet" selon votre config
THRESHOLD_SCORE = 60
EMOTION_INTERVAL = 10    # Nombre de frames entre les analyses d'émotion
DETECT_INTERVAL = 30     # Nombre de frames entre les détections de visages
TRACK_HISTORY = 5        # Historique des émotions à conserver

def create_kcf_tracker():
    """Crée un tracker KCF"""
    return cv2.TrackerKCF_create() if hasattr(cv2, 'TrackerKCF_create') else cv2.legacy.TrackerKCF_create()

def detect_faces_resized(frame, face_cascade, target_width=320):
    """Détection de visages avec redimensionnement pour performance"""
    h, w = frame.shape[:2]
    scale = w / target_width
    small = cv2.resize(frame, (target_width, int(h * target_width / w)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return [(int(x*scale), int(y*scale), int(w*scale), int(h*scale)) for (x,y,w,h) in faces]

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam.")
        return

    trackers = []  # Liste des trackers actifs
    emotions_history = []  # Historique des émotions
    frame_count = 0

    print("Lancement du suivi d'émotions. Appuyez sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Détection périodique de nouveaux visages
        if frame_count % DETECT_INTERVAL == 1:
            faces = detect_faces_resized(frame, face_cascade)
            
            for (x, y, w, h) in faces:
                # Vérifier les chevauchements avec les trackers existants
                overlap = any(
                    max(x, tx) < min(x+w, tx+tw) and max(y, ty) < min(y+h, ty+th)
                    for (tx, ty, tw, th) in [t["bbox"] for t in trackers]
                )
                
                if not overlap:
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Utilisation de votre fonction recognize_face existante
                    name, score = recognize_face(face_img)
                    
                    # Initialisation du tracker
                    tracker = create_kcf_tracker()
                    tracker.init(frame, (x, y, w, h))
                    
                    trackers.append({
                        "tracker": tracker,
                        "bbox": (x, y, w, h),
                        "name": name,
                        "last_emotion": None,
                        "last_recognized": frame_count
                    })
                    emotions_history.append(deque(maxlen=TRACK_HISTORY))

        # Mise à jour des trackers existants
        to_remove = []
        for i, tracker_data in enumerate(trackers):
            success, bbox = tracker_data["tracker"].update(frame)
            
            if not success:
                to_remove.append(i)
                continue
                
            x, y, w, h = map(int, bbox)
            tracker_data["bbox"] = (x, y, w, h)

            # Analyse d'émotion périodique
            if frame_count % EMOTION_INTERVAL == 0:
                face_img = frame[y:y+h, x:x+w]
                
                try:
                    # Analyse d'émotion avec DeepFace
                    result = DeepFace.analyze(face_img, actions=["emotion"], enforce_detection=False)
                    emotion = result[0]["dominant_emotion"]
                    tracker_data["last_emotion"] = emotion
                    emotions_history[i].append(emotion)
                    
                    # Log des résultats (à adapter selon vos besoins)
                    print(f"{datetime.now().strftime('%H:%M:%S')} | {tracker_data['name']} | Emotion: {emotion}")
                    
                except Exception as ex:
                    print(f"Erreur analyse émotion: {ex}")

        # Suppression des trackers perdus
        for i in sorted(to_remove, reverse=True):
            trackers.pop(i)
            emotions_history.pop(i)

        # Affichage
        for tracker_data in trackers:
            x, y, w, h = tracker_data["bbox"]
            
            # Couleur en fonction de la reconnaissance
            if tracker_data["name"] == "Inconnu":
                color = (0, 0, 255)  # Rouge pour inconnu
            elif tracker_data["name"] == "Erreur":
                color = (255, 0, 0)   # Bleu pour erreur
            else:
                color = (0, 255, 0)   # Vert pour reconnu
                
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Texte à afficher
            emotion = tracker_data["last_emotion"] or "Analyse..."
            display_text = f"{tracker_data['name']} | {emotion}"
            
            cv2.putText(frame, display_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Suivi Emotion + Reconnaissance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()