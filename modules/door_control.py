import serial
from config import serial_port, baud_rate

# esp32 = serial.Serial(serial_port, baud_rate)

def send_move_command():
    esp32.write(b"MOVE\n")
    esp32.write(b"OK\n")

def send_unknown_alert():
    esp32.write(b"INCONNU\n")

def send_error_alert():
    esp32.write(b"ERREUR\n")
