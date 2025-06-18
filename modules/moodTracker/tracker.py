import cv2
import time
import threading
from face_analysis import FaceAnalyzer
from voice_feedback import VoiceFeedback

class RealTimeMoodTracker:
    def __init__(self, max_faces=3):
        self.cap = cv2.VideoCapture(0)
        self.analyzer = FaceAnalyzer(max_faces=max_faces)
        self.voice = VoiceFeedback()
        self.face_results = []
        self.lock = threading.Lock()
        self.running = True
        self.analyzing = False

    def analyze_faces(self, frame):
        if self.analyzing:
            return
        
        self.analyzing = True

        def thread_task():
            results = self.analyzer.analyze(frame)
            with self.lock:
                self.face_results = results
            if not results:
                self.voice.say("Aucun élève détecté ou mouvement trop important.")
            else:
                for idx, res in enumerate(results):
                    emotion = res['dominant_emotion']
                    name = res.get("name", f"Face {idx + 1}")
                    self.voice.say(f"{name}, dominant emotion : {emotion}")
            self.analyzing = False

        threading.Thread(target=thread_task, daemon=True).start()

    def draw_annotations(self, frame):
        with self.lock:
            for idx, res in enumerate(self.face_results):
                region = res.get("region", {})
                if all(k in region for k in ("x", "y", "w", "h")):
                    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                    emotion = res["dominant_emotion"]
                    name = res.get("name", f"Face {idx + 1}")
                    label = f"{name}: {emotion}"
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def run(self):
        print("Suivi en direct (Q pour quitter)")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Erreur de capture.")
                break

            self.analyze_faces(frame)

            self.draw_annotations(frame)
            cv2.imshow("Suivi émotionnel en direct", frame)

            key = cv2.waitKey(1)
            if key in [ord('q'), 27]:
                self.running = False

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("Suivi arrêté.")
