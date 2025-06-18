from modules.face_detection import detect_faces
from modules.face_recognition import recognize_face
from modules.attendance_check import check_schedule
from modules.door_control import send_move_command, send_unknown_alert, send_error_alert
from modules.alert_system import say_message
from config import *
import cv2
import time
import pyttsx3
import csv
from datetime import datetime

saluted_names = set()
last_identity = None
last_sent_time = 0
cooldown_seconds = 120
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la webcam.")
    exit()

print("Reconnaissance en temps réel active. Appuie sur Q pour quitter.")

frame_count = 0
start_time = time.time()

tracked_faces = []  # [ {"coords": (x, y, w, h), "name": str, "last_seen": timestamp} ]
cooldown_time = 5  # secondes

log_file = "logs/access_log.csv"

# Créer le fichier de log s'il n'existe pas
with open(log_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    if file.tell() == 0:
        writer.writerow(["prenom", "date", "a_cours"])

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur de capture.")
        break

    frame_count += 1
    faces = detect_faces(frame)
    now = time.time()

    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        matched = False

        for face in tracked_faces:
            fx, fy, fw, fh = face["coords"]
            f_center = (fx + fw // 2, fy + fh // 2)
            if abs(center[0] - f_center[0]) < 50 and abs(center[1] - f_center[1]) < 50:
                matched = True
                if now - face["last_seen"] < cooldown_time:
                    name = face["name"]
                else:
                    face_img = frame[y:y+h, x:x+w]
                    recog_start = time.time()
                    name, score = recognize_face(face_img)
                    recog_duration = time.time() - recog_start
                    print(f"[INFO] Reconnaissance de {name} en {recog_duration:.2f} secondes")
                    face["name"] = name
                    face["last_seen"] = now
                face["coords"] = (x, y, w, h)
                break

        if not matched:
            face_img = frame[y:y+h, x:x+w]
            recog_start = time.time()
            name, score = recognize_face(face_img)
            recog_duration = time.time() - recog_start
            print(f"[INFO] Reconnaissance de {name} en {recog_duration:.2f} secondes")
            tracked_faces.append({"coords": (x, y, w, h), "name": name, "last_seen": now})

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if name != "Inconnu":
            if name not in saluted_names:
                cours, salle = check_schedule(name)
                if cours and salle:
                    message = f"Bonjour {name}, tu as cours de {cours} dans la salle {salle}"
                    say_message(message)
                    saluted_names.add(name)
                    last_identity = name
                    last_sent_time = now
                    send_move_command()

                    # Log entry
                    with open(log_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "oui"])
                else:
                    print(f"{name} reconnu, mais pas de cours actuellement.")
                    # Log entry
                    with open(log_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "non"])
        else:
            if last_identity != "Inconnu" or (now - last_sent_time > cooldown_seconds):
                send_unknown_alert()
                last_identity = "Inconnu"
                last_sent_time = now

    cv2.imshow("Reconnaissance en temps réel [Q pour quitter]", frame)
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
