import cv2
<<<<<<< HEAD

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces
=======
import os

def capture_faces(person_name, output_dir="../dataset", num_images=10):
    # Créer le dossier pour la personne
    person_path = os.path.join(output_dir, person_name)
    os.makedirs(person_path, exist_ok=True)

    # Chargement du détecteur de visages HaarCascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam.")
        return

    print(f"Capture de {num_images} images pour {person_name}")
    print("Appuie sur ESPACE pour capturer une image, ou Q pour quitter.")

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Erreur de capture.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Dessiner les rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Webcam - Appuie sur [ESPACE] pour capturer | [Q] pour quitter", frame)
        key = cv2.waitKey(1)

        if key == ord(' '):  # ESPACE
            if len(faces) == 0:
                print("Aucun visage détecté.")
                continue

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (224, 224))  # Taille compatible DeepFace
                img_path = os.path.join(person_path, f"{person_name}_{count + 1}.jpg")
                cv2.imwrite(img_path, face_img)
                print(f"Image sauvegardée : {img_path}")
                count += 1
                break  # Une seule capture par pression

        elif key == ord('q') or key == 27:
            print("Capture interrompue.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Capture terminée.")

if __name__ == "__main__":
    name = input("Nom de la personne à enregistrer : ").strip()
    capture_faces(person_name=name)
>>>>>>> 8e24f5e7237c6f2b52bef32669bedf44e16f179e
