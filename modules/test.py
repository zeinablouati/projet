import cv2
import os
import time
import csv
import numpy as np
from collections import deque
from deepface import DeepFace

def create_kcf_tracker():
    if hasattr(cv2, 'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    else:
        return cv2.legacy.TrackerKCF_create()

def recognize_name(face_img, database_path, model_name, threshold_score):
    """
    Renvoie un nom à partir d'une image de visage (ROI).
    """
    try:
        results = DeepFace.find(
            img_path = face_img,
            db_path = database_path,
            model_name = model_name,
            enforce_detection = False
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
                return os.path.basename(os.path.dirname(identity_path))
    except Exception:
        pass
    return "Inconnu"

def main():
    # --- CONFIG ---
    database_path = "../dataset"
    model_name = "VGG-Face"
    threshold_score = 60
    window_size = 30  # frames pour calcul écarts-type
    vote_len = 5      # sliding window length
    vote_thresh = 3   # nombre de True pour voter "Distrait"
    detect_interval = 60
    max_faces = 10
    log_filename = "concentration_log.csv"

    # --- INITIALISATION CSV LOG ---
    csvfile = open(log_filename, mode='w', newline='', encoding='utf-8')
    writer = csv.writer(csvfile)
    writer.writerow(["timestamp", "name", "status"])

    # --- CHARGEMENT HAAR ---
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: impossible d'ouvrir la caméra.")
        return

    # --- STRUCTURES DE SUIVI ---
    tracked = []
    # chaque élément : {
    #   "name": str,
    #   "tracker": TrackerKCF,
    #   "bbox": (x,y,w,h),
    #   "history": deque(maxlen=window_size),
    #   "votes": deque(maxlen=vote_len)
    # }

    frame_count = 0

    print("Appuyez sur 'q' pour quitter…")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        h_full, w_full = frame.shape[:2]

        # ----- 1. (Re)détection périodique -----
        if frame_count % detect_interval == 1:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            # tri par aire decroissante
            faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[:max_faces]

            for (x, y, w, h) in faces:
                # voir si ce visage existe déjà (centre proche)
                cx, cy = x + w//2, y + h//2
                exists = False
                for entry in tracked:
                    xo, yo, wo, ho = entry["bbox"]
                    ocx, ocy = xo + wo//2, yo + ho//2
                    if abs(cx-ocx)<50 and abs(cy-ocy)<50:
                        exists = True
                        break
                if exists:
                    continue

                # sinon, reconnaître et ajouter
                face_img = frame[y:y+h, x:x+w]
                name = recognize_name(face_img, database_path, model_name, threshold_score)

                trk = create_kcf_tracker()
                trk.init(frame, (x,y,w,h))
                tracked.append({
                    "name": name,
                    "tracker": trk,
                    "bbox": (x,y,w,h),
                    "history": deque(maxlen=window_size),
                    "votes": deque(maxlen=vote_len)
                })

        # ----- 2. Mise à jour trackers et historique -----
        to_remove = []
        for i, entry in enumerate(tracked):
            ok, bbox = entry["tracker"].update(frame)
            if not ok:
                to_remove.append(i)
                continue
            x, y, w, h = map(int, bbox)
            entry["bbox"] = (x, y, w, h)
            cx, cy = x + w//2, y + h//2
            entry["history"].append((cx, cy))

        # suppression trackers HS
        for i in sorted(to_remove, reverse=True):
            tracked.pop(i)

        # ----- 3. Calcul de concentration & logs -----
        now = time.time()
        for entry in tracked:
            name = entry["name"]
            hist = entry["history"]
            votes = entry["votes"]

            status = "Collecte…"
            color = (0,255,255)
            if len(hist) >= window_size:
                arr = np.array(hist)
                std_x = np.std(arr[:,0])
                std_y = np.std(arr[:,1])
                # flag si mouvement > seuil fixe (disons 15 px)
                flag = (std_x > 15.0) or (std_y > 15.0)
                votes.append(flag)
                # sliding-window vote
                if len(votes) >= vote_len:
                    if sum(votes) >= vote_thresh:
                        status = "Distrait"
                        color = (0,0,255)
                    else:
                        status = "Concentre"
                        color = (0,255,0)
                else:
                    status = "Trop peu de données"

                # log CSV
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
                writer.writerow([timestamp, name, status])
                csvfile.flush()

            # --- Affichage ---
            x,y,w,h = entry["bbox"]
            cv2.rectangle(frame, (x,y),(x+w,y+h), color, 2)
            cv2.putText(frame, f"{name}: {status}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Reconnaissance + Concentration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    csvfile.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
