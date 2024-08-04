from openvoicechat.tts.tts_xtts import Mouth_xtts
from openvoicechat.llm.llm_gpt import Chatbot_gpt
from openvoicechat.stt.stt_hf import Ear_hf
from openvoicechat.utils import run_chat
from dotenv import load_dotenv
import os
from prompts import appointment_prompt as prompt
from prompts import tools
import torch
import pandas as pd

load_dotenv()

doctor_schedule = pd.read_csv("doctors_schedule.csv")


def get_available_times(doctor):
    return ' ,'.join(
        doctor_schedule[(doctor_schedule['Doctor'] == doctor) & (doctor_schedule['Booked'] == False)]['Time'].values)


def make_appointment(doctor, time):
    print("Doctor:", doctor)
    print("Time:", time)
    booked = doctor_schedule[(doctor_schedule['Doctor'] == doctor) &
                             (doctor_schedule['Time'] == time)]['Booked'].values[0]
    print("Making appointment at time:", time)
    if booked:
        return "Sorry, that time is already booked. Please choose another time."
    else:
        doctor_schedule.loc[(doctor_schedule['Doctor'] == doctor) &
                            (doctor_schedule['Time'] == time), 'Booked'] = True
        doctor_schedule.to_csv("doctors_schedule.csv", index=False)
        return "Appointment booked successfully"


func_utterance = {
    'make_appointment': ["Let me see if I can book that time for you.",
                         "Let me make that booking.",
                         "Let me get back to you with the booking details."],
    'get_available_times': ["Let me check the available times for you.",
                            "Let me see what times are available.",
                            "Let me check the schedule for you."]}

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading models... ', device)
    ear = Ear_hf(silence_seconds=1.5,
                 device=device)
    load_dotenv()
    # chatbot = AppointmentChatbot(sys_prompt=prompt)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    chatbot = Chatbot_gpt(sys_prompt=prompt, api_key=OPENAI_API_KEY,
                          tool_choice='auto', tools=tools,
                          tool_utterances=func_utterance,
                          functions={'make_appointment': make_appointment,
                                     'get_available_times': get_available_times})

    mouth = Mouth_xtts(device=device,
                       model_id='tts_models/multilingual/multi-dataset/xtts_v2',
                       speaker='Ana Florence')

    _ = mouth.run_tts('Hello, I am your assistant. How can I help you today?')  # warm up the model
    ear.transcribe(_)  # warm up the model

    run_chat(mouth, ear, chatbot, verbose=True, stopping_criteria=lambda x: '[END]' in x)
