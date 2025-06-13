import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def say_message(message):
    print(message)
    engine.say(message)
    engine.runAndWait()
