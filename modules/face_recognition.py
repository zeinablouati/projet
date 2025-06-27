from deepface import DeepFace
import os
from config import database_path, model_name, threshold_score
import cv2
import time
import traceback

def recognize_face(face_img):
    try:
        # Vérifier la qualité de l'image avant traitement
        if face_img is None:
            print("ERREUR: Image None reçue")
            return "Erreur", 0
            
        if face_img.size == 0:
            print("ERREUR: Image vide reçue")
            return "Erreur", 0
            
        if face_img.shape[0] < 30 or face_img.shape[1] < 30:
            print(f"ERREUR: Image trop petite {face_img.shape}")
            return "Erreur", 0
            
        print(f"INFO: Traitement image de taille {face_img.shape}")
        
        # Sauvegarder temporairement l'image pour le débogage
        temp_path = "temp_face.jpg"
        success = cv2.imwrite(temp_path, face_img)
        
        if not success:
            print("ERREUR: Impossible de sauvegarder l'image temporaire")
            return "Erreur", 0
            
        # Vérifier que le fichier existe
        if not os.path.exists(temp_path):
            print("ERREUR: Fichier temporaire non créé")
            return "Erreur", 0
            
        print(f"INFO: Fichier temporaire créé: {temp_path}")
        
        # Vérifier que la base de données existe
        if not os.path.exists(database_path):
            print(f"ERREUR: Base de données introuvable: {database_path}")
            return "Erreur", 0
            
        print(f"INFO: Base de données trouvée: {database_path}")
        
        # Compter les images dans la base de données
        image_count = 0
        for root, dirs, files in os.walk(database_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_count += 1
        print(f"INFO: {image_count} images trouvées dans la base de données")
        
        if image_count == 0:
            print("ERREUR: Aucune image dans la base de données")
            return "Erreur", 0
        
        results = DeepFace.find(
            img_path=temp_path,
            db_path=database_path,
            model_name=model_name,
            enforce_detection=False,
            silent=True
        )
        
        print(f"INFO: DeepFace.find terminé, résultats: {type(results)}")
        
        # Nettoyer le fichier temporaire
        try:
            os.remove(temp_path)
        except:
            pass
        
        if not results or len(results) == 0:
            print("INFO: Aucun résultat retourné par DeepFace")
            return "Inconnu", 0
            
        df = results[0]
        print(f"INFO: DataFrame reçu, shape: {df.shape}")
        
        if df.empty:
            print("INFO: DataFrame vide")
            return "Inconnu", 0
            
        df = df.sort_values(by="distance", ascending=True).reset_index(drop=True)
        best = df.iloc[0]
        identity_path = best["identity"]
        distance = best["distance"]
        threshold = best.get("threshold", 0.4)
        score = (1 - distance / threshold) * 100
        
        print(f"INFO: Meilleur match - Distance: {distance}, Score: {score:.2f}%, Seuil: {threshold_score}")
        
        if score >= threshold_score:
            name = os.path.basename(os.path.dirname(identity_path))
            print(f"SUCCESS: Personne reconnue: {name}")
            return name, score
        else:
            print(f"INFO: Score trop faible ({score:.2f}% < {threshold_score}%)")
            return "Inconnu", score
            
    except Exception as e:
        print(f"EXCEPTION DÉTAILLÉE: {type(e).__name__}: {str(e)}")
        print("TRACEBACK COMPLET:")
        traceback.print_exc()
        
        # Essayer de nettoyer le fichier temporaire même en cas d'erreur
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass
            
        return "Erreur", 0

def recognize_faces_realtime(database_path="../dataset", model_name="VGG-Face", threshold_score=60):
    print(f"DÉMARRAGE avec paramètres:")
    print(f"- Database: {database_path}")
    print(f"- Model: {model_name}")
    print(f"- Threshold: {threshold_score}")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERREUR: Impossible d'ouvrir la caméra")
        return
        
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    if face_cascade.empty():
        print("ERREUR: Impossible de charger le classificateur de visages")
        return

    tracked_faces = []  # [ {x, y, w, h, name, last_seen} ]
    cooldown_time = 5  # secondes
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERREUR: Impossible de lire le frame")
            break
            
        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        print(f"Frame {frame_count}: {len(faces)} visage(s) détecté(s)")

        current_time = time.time()

        for i, (x, y, w, h) in enumerate(faces):
            print(f"Traitement visage {i+1}/{len(faces)} - Position: ({x},{y},{w},{h})")
            
            center = (x + w // 2, y + h // 2)
            matched = False

            # Essayer d'associer avec un visage déjà reconnu
            for face in tracked_faces:
                fx, fy, fw, fh = face["coords"]
                f_center = (fx + fw // 2, fy + fh // 2)

                if abs(center[0] - f_center[0]) < 50 and abs(center[1] - f_center[1]) < 50:
                    matched = True
                    print(f"Visage associé à un visage existant")
                    
                    if current_time - face["last_seen"] < cooldown_time:
                        name = face["name"]
                        print(f"Utilisation du nom en cache: {name}")
                    else:
                        print(f"Cooldown expiré, nouvelle reconnaissance...")
                        face_img = frame[y:y+h, x:x+w]
                        name, score = recognize_face(face_img)

                        face["name"] = name
                        face["last_seen"] = current_time

                    face["coords"] = (x, y, w, h)
                    break

            # Si pas de correspondance, on ajoute un nouveau visage
            if not matched:
                print(f"Nouveau visage détecté, reconnaissance...")
                face_img = frame[y:y+h, x:x+w]
                name, score = recognize_face(face_img)

                tracked_faces.append({
                    "coords": (x, y, w, h),
                    "name": name,
                    "last_seen": current_time
                })

            # Affichage
            color = (0, 255, 0) if name not in ["Inconnu", "Erreur"] else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Reconnaissance en temps réel [Q pour quitter]", frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces_realtime()