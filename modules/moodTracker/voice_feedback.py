import pyttsx3
import threading
from queue import Queue

class VoiceFeedback:
    def __init__(self):
        self.queue = Queue()
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 160)
        threading.Thread(target=self._process_queue, daemon=True).start()

    def say(self, message):
        self.queue.put(message)

    def _process_queue(self):
        while True:
            text = self.queue.get()
            self.engine.say(text)
            self.engine.runAndWait()