import cv2
import numpy as np
import time
from datetime import datetime
from collections import deque
from deepface import DeepFace
from modules.face_recognition import recognize_face  # Import de votre fonction existante

# === Configuration du logger ===
class DataLogger:
    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = fieldnames
        # Initialisation du fichier (implémentation simplifiée)
        
    def log(self, **kwargs):
        # Implémentation de la journalisation
        pass

logger = DataLogger(
    filename="concentration_report.csv",
    fieldnames=["tracker", "status", "std_x", "std_y"]
)

summary_logger = DataLogger(
    filename="summary_report.csv",
    fieldnames=[
        "Etudiant",
        "period_start",
        "period_end",
        "num_distrait",
        "num_concentre",
        "pct_concentre"
    ]
)

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
    # Initialisation
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la caméra.")
        return

    # Calibration du seuil
    calib = []
    print("Calibration : restez immobile 2 s pour mesurer le bruit...")
    t0 = time.time()
    while time.time() - t0 < 2.0:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces_resized(frame, face_cascade)
        if faces:
            x, y, w, h = faces[0]
            calib.append((x + w//2, y + h//2))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    SEUIL_STD = max(np.std(np.array(calib)[:, 0]), np.std(np.array(calib)[:, 1])) * 1.5 if len(calib) >= 30 else 15.0
    print(f"Calibration terminée : SEUIL_STD = {SEUIL_STD:.2f} pixels")

    # Initialisation des structures de suivi
    trackers = []
    history_list = []
    status_history_list = []
    window_size = 30
    frame_count = 0
    detect_interval = 60
    max_faces = 5

    # Compteurs de périodicité
    period_start = time.time()
    distrait_count = 0
    sum_stdx = 0.0
    sum_stdy = 0.0
    event_count = 0

    print("Suivi démarré. Appuyez sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Détection périodique de nouveaux visages
        if frame_count % detect_interval == 1:
            faces = detect_faces_resized(frame, face_cascade)
            if faces:
                faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[:max_faces]
                for (x, y, w, h) in faces:
                    # Vérifier les chevauchements
                    overlap = any(
                        max(x, tx) < min(x+w, tx+tw) and max(y, ty) < min(y+h, ty+th)
                        for (tx, ty, tw, th) in [t["bbox"] for t in trackers]
                    )
                    if not overlap:
                        face_img = frame[y:y+h, x:x+w]
                        
                        # Utilisation de votre fonction recognize_face
                        name, score = recognize_face(face_img)
                        
                        # Initialisation du tracker
                        tracker = create_kcf_tracker()
                        tracker.init(frame, (x, y, w, h))
                        
                        trackers.append({
                            "tracker": tracker,
                            "bbox": (x, y, w, h),
                            "name": name,
                            "score": score
                        })
                        history_list.append(deque(maxlen=window_size))
                        status_history_list.append(deque(maxlen=5))

        # Mise à jour des trackers existants
        to_remove = []
        for i, tracker_data in enumerate(trackers):
            success, bbox = tracker_data["tracker"].update(frame)
            
            if not success:
                to_remove.append(i)
                continue
                
            x, y, w, h = map(int, bbox)
            tracker_data["bbox"] = (x, y, w, h)
            cx, cy = x + w//2, y + h//2
            history_list[i].append((cx, cy))

        # Suppression des trackers perdus
        for i in sorted(to_remove, reverse=True):
            trackers.pop(i)
            history_list.pop(i)
            status_history_list.pop(i)

        # Analyse et affichage
        for i, tracker_data in enumerate(trackers):
            x, y, w, h = tracker_data["bbox"]
            
            # Dessin du rectangle et du point central
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if history_list[i]:
                cx, cy = history_list[i][-1]
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            status = "Collecte"
            color = (0, 255, 255)  # Jaune
            std_x = std_y = 0.0

            if len(history_list[i]) >= window_size:
                arr = np.array(history_list[i])
                std_x = np.std(arr[:, 0])
                std_y = np.std(arr[:, 1])

                # Détection de mouvement
                flag = (std_x > SEUIL_STD) or (std_y > SEUIL_STD)
                status_history_list[i].append(flag)

                if len(status_history_list[i]) >= 5:
                    if sum(status_history_list[i]) >= 3:
                        status, color = "Distrait", (0, 0, 255)  # Rouge
                    else:
                        status, color = "Concentre", (0, 255, 0)  # Vert

                # Journalisation
                logger.log(
                    tracker=tracker_data["name"],
                    status=status,
                    std_x=f"{std_x:.2f}",
                    std_y=f"{std_y:.2f}"
                )
                
                # Mise à jour des compteurs
                event_count += 1
                sum_stdx += std_x
                sum_stdy += std_y
                if status == "Distrait":
                    distrait_count += 1

            # Affichage du statut
            label = f"{tracker_data['name']} ({tracker_data['score']:.1f}%) : {status}"
            cv2.putText(frame, label, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Gestion des périodes de 2 minutes
        now = time.time()
        if now - period_start >= 120.0 and event_count > 0:
            period_end = now
            num_concentre = event_count - distrait_count
            pct_concentre = (num_concentre / event_count) * 100

            summary_logger.log(
                Etudiant=trackers[0]["name"] if trackers else "Inconnu",
                period_start=datetime.fromtimestamp(period_start).strftime("%Y-%m-%d %H:%M:%S"),
                period_end=datetime.fromtimestamp(period_end).strftime("%Y-%m-%d %H:%M:%S"),
                num_distrait=distrait_count,
                num_concentre=num_concentre,
                pct_concentre=f"{pct_concentre:.1f}"
            )

            # Réinitialisation des compteurs
            period_start = now
            distrait_count = 0
            sum_stdx = 0.0
            sum_stdy = 0.0
            event_count = 0

        # Affichage final
        cv2.imshow("Suivi Attention + Reconnaissance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()