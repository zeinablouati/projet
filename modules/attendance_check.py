import pandas as pd
from datetime import datetime
from config import edt_path

edt_df = pd.read_csv(edt_path)

def check_schedule(name):
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M")
    user_courses = edt_df[
        (edt_df['nom'].str.lower() == name.lower()) &
        (edt_df['date'] == current_date)
    ]
    for _, row in user_courses.iterrows():
        if row['heure_debut'] <= current_time <= row['heure_fin']:
            return row['cours'], row['salle']
    return None, None
