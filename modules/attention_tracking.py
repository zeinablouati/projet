import cv2
import numpy as np
import time
<<<<<<< HEAD
from datetime import datetime
from collections import deque
from data_logger import recognize_name, DataLogger

#1) Configuration du logger ===
=======
import datetime
from collections import deque
import os
from deepface import DeepFace
from data_logger import recognize_name, DataLogger

# === 1) Configuration du logger ===
>>>>>>> 8e24f5e7237c6f2b52bef32669bedf44e16f179e
logger = DataLogger(
    filename="concentration_report.csv",
    fieldnames=["tracker", "status", "std_x", "std_y"]
)
summary_logger = DataLogger(
    filename="summary_report.csv",
<<<<<<< HEAD
    fieldnames=[
        "Etudiant",
        "period_start",
        "period_end",
        "num_distrait",
        "num_concentre",
        "pct_concentre"
    ]
)



=======
    fieldnames=["period_start", "period_end", "num_distrait", "avg_std_x", "avg_std_y"]
)


>>>>>>> 8e24f5e7237c6f2b52bef32669bedf44e16f179e
def create_kcf_tracker():
    """
    Crée et retourne un tracker KCF.
    Essaie d'abord cv2.TrackerKCF_create, sinon cv2.legacy.TrackerKCF_create.
    """
    if hasattr(cv2, 'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    else:
        return cv2.legacy.TrackerKCF_create()

def detect_faces_resized(frame, face_cascade, target_width=320):
    """
    Détecte les visages sur une version réduite de frame (largeur=target_width),
    puis rééchelle les bounding boxes pour la résolution initiale.
    """
    h_full, w_full = frame.shape[:2]
    scale = w_full / target_width
    small_h = int(h_full * target_width / w_full)
    small = cv2.resize(frame, (target_width, small_h))
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces_small = face_cascade.detectMultiScale(gray_small, 1.3, 5)

    faces_full = []
    for (x, y, w, h) in faces_small:
        faces_full.append((
            int(x * scale),
            int(y * scale),
            int(w * scale),
            int(h * scale)
        ))
    return faces_full

def main():
    # 1. Chargement du classifieur Haar
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # 2. Ouverture de la webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la caméra.")
        return

    # 3. Calibration du seuil sur 2 secondes
    calib = []
    print("Calibration : restez immobile 2 s pour mesurer le bruit…")
    t0 = time.time()
    while time.time() - t0 < 2.0:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces_resized(frame, face_cascade, target_width=320)
        if faces:
            x, y, w, h = faces[0]
            calib.append((x + w//2, y + h//2))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if len(calib) >= 30:
        arr0 = np.array(calib)
        base_x = np.std(arr0[:, 0])
        base_y = np.std(arr0[:, 1])
        SEUIL_STD = max(base_x, base_y) * 1.5
    else:
        SEUIL_STD = 15.0

    print(f"Calibration terminée : SEUIL_STD = {SEUIL_STD:.2f} pixels")

    # 4. Initialisation des structures de suivi
    trackers = []
    history_list = []
    status_history_list = []
    window_size = 30
    # 5. Paramètres périodiques
    frame_count     = 0
    detect_interval = 60
    max_faces = 5
      # === NOUVEAU : compteurs de périodicité 2 minutes ===
    period_start   = time.time()
    distrait_count = 0
    sum_stdx       = 0.0
    sum_stdy       = 0.0
    event_count    = 0

    print("Suivi démarré. Appuyez sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # 5a. Redétection périodique
        if frame_count % detect_interval == 1:
            faces = detect_faces_resized(frame, face_cascade, target_width=320)
            if faces:
                faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[:max_faces]
                for (x, y, w, h) in faces:
                    # vérifier chevauchement
                    overlap = False
                    for e in trackers:
                        xo, yo, wo, ho = e["bbox"]
                        ix = max(0, min(x+w, xo+wo) - max(x, xo))
                        iy = max(0, min(y+h, yo+ho) - max(y, yo))
                        if ix*iy > 0:
                            overlap = True
                            break
                    if overlap:
                        continue
                    # ajouter un nouveau tracker
                    trk = create_kcf_tracker()
                    trk.init(frame, (x, y, w, h))
                  # Nouveau : on découpe le visage et on appelle la reconnaissance
                    face_img = frame[y:y+h, x:x+w]
                    name = recognize_name(
                        face_img,
                        db_path="../dataset",
                        model_name="VGG-Face",
                        threshold_score=60
                    )

                    # Ensuite on crée le tracker et on stocke le nom
                    tracker = create_kcf_tracker()
                    tracker.init(frame, (x, y, w, h))
                    trackers.append({
                        "tracker": tracker,
                        "bbox": (x, y, w, h),
                        "name": name
                    })
                    history_list.append(deque(maxlen=window_size))
                    status_history_list.append(deque(maxlen=5))

                    history_list.append(deque(maxlen=window_size))
                    status_history_list.append(deque(maxlen=5))

        # 5b. Mise à jour des trackers
        to_remove = []
        for i, e in enumerate(trackers):
            ok, bbox = e["tracker"].update(frame)
            if not ok:
                to_remove.append(i)
                continue
            x, y, w, h = map(int, bbox)
            e["bbox"] = (x, y, w, h)
            cx, cy = x + w//2, y + h//2
            history_list[i].append((cx, cy))
        # suppression des trackers échoués
        for i in sorted(to_remove, reverse=True):
            trackers.pop(i)
            history_list.pop(i)
            status_history_list.pop(i)

        # 5c. Affichage, classification et LOG
        for i, e in enumerate(trackers):
            x, y, w, h = e["bbox"]
            # dessin du rectangle et du point central
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if history_list[i]:
                cx, cy = history_list[i][-1]
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            status = "Collecte"
            color = (0, 255, 255)
            std_x = std_y = 0.0

            if len(history_list[i]) >= window_size:
                arr = np.array(history_list[i])
                std_x = np.std(arr[:, 0])
                std_y = np.std(arr[:, 1])

                # détection de secousse
                flag = (std_x > SEUIL_STD) or (std_y > SEUIL_STD)
                status_history_list[i].append(flag)

                last5 = list(status_history_list[i])
                count_true = sum(last5)

                if len(status_history_list[i]) >= 5:
                    if count_true >= 3:
                        status, color = "Distrait", (0, 0, 255)
                    else:
                        status, color = "Concentre", (0, 255, 0)
                else:
                    status, color = "Trop peu de données", (0, 255, 255)

                # === LOG ===
                logger.log(
                    tracker=e["name"],
                    status=status,
                    std_x=f"{std_x:.2f}",
                    std_y=f"{std_y:.2f}"
                )
                event_count += 1
                sum_stdx += std_x
                sum_stdy += std_y
                if status == "Distrait":
                     distrait_count += 1

            # affichage du statut
            label = f"{e['name']} : {status}"
            cv2.putText(frame, label, (x, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Si la période arrive à 120 s, on envoie le résumé
        now = time.time()
        if now - period_start >= 120.0 and event_count > 0:
            period_end = now
            avg_x = sum_stdx / event_count
            avg_y = sum_stdy / event_count


<<<<<<< HEAD
                # calcul des nouvelles métriques
            num_concentre = event_count - distrait_count
            pct_concentre = (num_concentre / event_count) * 100

            summary_logger.log(
                    Etudiant=  e["name"],
                    period_start   = datetime.fromtimestamp(period_start).strftime("%Y-%m-%d %H:%M:%S"),
                    period_end     = datetime.fromtimestamp(period_end).strftime("%Y-%m-%d %H:%M:%S"),
                    num_distrait   = distrait_count,
                    num_concentre  = num_concentre,
                    pct_concentre  = f"{pct_concentre:.1f}"
                )

=======
            # Écrire la ligne de résumé
            summary_logger.log(
                period_start = datetime.fromtimestamp(period_start).strftime("%Y-%m-%d %H:%M:%S"),
                period_end   = datetime.fromtimestamp(period_end).strftime("%Y-%m-%d %H:%M:%S"),
                num_distrait = distrait_count,
                avg_std_x    = f"{avg_x:.2f}",
                avg_std_y    = f"{avg_y:.2f}"
            )
>>>>>>> 8e24f5e7237c6f2b52bef32669bedf44e16f179e

            # Remise à zéro pour la prochaine période
            period_start   = now
            distrait_count = 0
            sum_stdx       = 0.0
            sum_stdy       = 0.0
            event_count    = 0

        # 6. Affichage final
        cv2.imshow("Multi-Tracking + Concentration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
   