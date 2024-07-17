import pandas as pd
import numpy as np

# Assuming the same doctors and roles from `data.py`
doctors = ["Doctor John", "Doctor Jane", "Doctor Smith"]


def get_available_times():
    booked = [bool(x) for x in np.random.randint(0, 2, 13)]
    times = []
    for hour, is_booked in enumerate(booked, start=9):
        formatted_time = f"{hour if hour <= 12 else hour - 12} {'am' if hour < 12 or hour == 24 else 'pm'}"
        times.append((formatted_time, is_booked))
    return times


def create_doctors_schedule_csv():
    times_booked = get_available_times()
    schedule_data = []

    for doctor in doctors:
        for time, is_booked in times_booked:
            schedule_data.append({"Doctor": doctor, "Time": time, "Booked": is_booked})

    schedule_df = pd.DataFrame(schedule_data)
    schedule_df.to_csv("doctors_schedule.csv", index=False)


create_doctors_schedule_csv()
