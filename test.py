import cv2
import time
import os
import csv
from datetime import datetime
from deepface import DeepFace

# Import des modules personnalisés
from modules.alert_system import say_message
from modules.attendance_check import check_schedule
from modules.attention_tracking import main as track_attention
from modules.data_logger import DataLogger
from modules.door_control import send_move_command, send_unknown_alert, send_error_alert
from modules.emotion_tracking import main as track_emotion
from modules.face_detection import detect_faces, capture_faces
from modules.face_recognition import recognize_face  # Votre fonction personnalisée
from modules.train_faces import capture_faces as train_new_face

# Configuration (identique)
DATABASE_PATH = "../dataset"
MODEL_NAME = "VGG-Face"
THRESHOLD_SCORE = 60
COOLDOWN_SECONDS = 120
LOG_FILE = "logs/access_log.csv"

class FaceTracker:
    def __init__(self):
        self.cap = None
        self.saluted_names = set()
        self.last_identity = None
        self.last_sent_time = 0
        self.initialize_camera()

    def initialize_camera(self):
        """Initialisation caméra identique"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            print("Erreur : impossible d'ouvrir la webcam.")
            exit()

    def check_existing_names(self):
        """Version strictement identique"""
        if not os.path.exists(DATABASE_PATH):
            return []
        return [name for name in os.listdir(DATABASE_PATH) 
                if os.path.isdir(os.path.join(DATABASE_PATH, name))]

    def presence_mode(self):
        """Mode présence avec intégration fluide des autres modules"""
        print("Mode présence activé - Reconnaissance en cours...")
        tracked_faces = []
        cooldown_time = 5

        # Initialisation fichier log
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["prenom", "date", "a_cours"])

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Erreur de capture.")
                break

            faces = detect_faces(frame)
            now = time.time()

            for (x, y, w, h) in faces:
                center = (x + w//2, y + h//2)
                matched = False

                # Tracking avec reconnaissance connectée
                for face in tracked_faces:
                    fx, fy, fw, fh = face["coords"]
                    f_center = (fx + fw//2, fy + fh//2)
                    if abs(center[0] - f_center[0]) < 50 and abs(center[1] - f_center[1]) < 50:
                        matched = True
                        if now - face["last_seen"] < cooldown_time:
                            name = face["name"]
                        else:
                            face_img = frame[y:y+h, x:x+w]
                            name, score = recognize_face(face_img)  # Appel connecté
                            face.update({
                                "name": name,
                                "last_seen": now
                            })
                        face["coords"] = (x, y, w, h)
                        break

                if not matched:
                    face_img = frame[y:y+h, x:x+w]
                    name, score = recognize_face(face_img)  # Appel connecté
                    tracked_faces.append({
                        "coords": (x, y, w, h),
                        "name": name,
                        "last_seen": now
                    })

                # Affichage avec score si disponible
                display_name = f"{name} ({score:.1f}%)" if name not in ["Inconnu", "Erreur"] and 'score' in locals() else name
                color = (0, 255, 0) if name not in ["Inconnu", "Erreur"] else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, display_name, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Logique métier strictement identique
                if name != "Inconnu":
                    if name not in self.saluted_names:
                        cours, salle = check_schedule(name)
                        if cours and salle:
                            say_message(f"Bonjour {name}, tu as cours de {cours} dans la salle {salle}")
                        self.saluted_names.add(name)
                        self.last_identity = name
                        self.last_sent_time = now

                        with open(LOG_FILE, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "oui" if cours else "non"])
                else:
                    if self.last_identity != "Inconnu" or (now - self.last_sent_time > COOLDOWN_SECONDS):
                        self.last_identity = "Inconnu"
                        self.last_sent_time = now

            cv2.imshow("Reconnaissance en temps réel [Q pour quitter]", frame)
            key = cv2.waitKey(1)
            
            if key == ord('q') or key == 27:
                break
            elif key == ord('a'):  # Basculer vers suivi d'attention
                self.cap.release()
                cv2.destroyAllWindows()
                track_attention()
                self.initialize_camera()
            elif key == ord('e'):  # Basculer vers suivi d'émotion
                self.cap.release()
                cv2.destroyAllWindows()
                track_emotion()
                self.initialize_camera()

    def registration_mode(self):
        """Mode enregistrement strictement identique"""
        existing_names = self.check_existing_names()
        print(f"\nNoms déjà enregistrés: {', '.join(existing_names) if existing_names else 'Aucun'}")
        
        while True:
            new_name = input("\nEntrez le nom de la personne à enregistrer (ou 'q' pour quitter): ").strip()
            
            if new_name.lower() == 'q':
                break
                
            if new_name in existing_names:
                print(f"Erreur: Le nom '{new_name}' existe déjà. Choisissez un autre nom.")
                continue
                
            if not new_name:
                print("Erreur: Le nom ne peut pas être vide.")
                continue
                
            print(f"\nDébut de l'enregistrement pour {new_name}...")
            train_new_face(new_name)
            existing_names = self.check_existing_names()

def main_menu():
    """Menu principal avec gestion améliorée des ressources"""
    tracker = FaceTracker()
    
    try:
        while True:
            print("\n" + "="*50)
            print("SYSTEME DE RECONNAISSANCE FACIALE".center(50))
            print("="*50)
            print("1. Mode Présence (reconnaissance)")
            print("2. Mode Enregistrement (nouveaux visages)")
            print("3. Suivi d'Attention")
            print("4. Suivi d'Émotion")
            print("5. Quitter")
            
            choice = input("\nVotre choix (1-5): ").strip()
            
            if choice == '1':
                tracker.presence_mode()
            elif choice == '2':
                tracker.registration_mode()
            elif choice == '3':
                tracker.cap.release()
                cv2.destroyAllWindows()
                track_attention()
                tracker.initialize_camera()
            elif choice == '4':
                tracker.cap.release()
                cv2.destroyAllWindows()
                track_emotion()
                tracker.initialize_camera()
            elif choice == '5':
                print("Fermeture du système...")
                break
            else:
                print("Choix invalide. Veuillez entrer un nombre entre 1 et 5.")
    finally:
        if tracker.cap is not None:
            tracker.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_menu()