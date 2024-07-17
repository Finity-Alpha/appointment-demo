import pandas as pd
import numpy as np
import os

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


#if doctors_schedule.csv does not exist, create it
if not os.path.exists("doctors_schedule.csv"):
    create_doctors_schedule_csv()

from openvoicechat.tts.tts_xtts import Mouth_xtts
from openvoicechat.llm.base import BaseChatbot
from openvoicechat.stt.stt_hf import Ear_hf
from openvoicechat.utils import run_chat
from dotenv import load_dotenv
from openai import OpenAI
import torch

prompt = '''
You are an appointment setter for the city hospital. You will be in a call with a client.
Appointments are available from 9 A M to 9 P M.
Ask the client for the time they would like to book an appointment and help them find
a suitable time.
Always say A M instead of AM or am when referring to the time, same with pm.
Always say doctor instead of Dr. when referring to the doctor.
Before ending the call, ask if the user needs any further assistance.
Output [END] when the conversation is over.
Keep your responses short and concise. 
The following are the doctors and their roles:
Doctor John - ENT Specialist
Doctor Jane - Orthopedic
Doctor Smith - Dentist
'''

tools = [
    {
        "type": "function",
        "function": {
            "name": "make_appointment",
            "description": "Set the appointment for the given time",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "integer",
                        "description": "The time to make the appointment at. Times are in the 24 hour format, "
                                       "the function only takes the hour.",
                    },
                    "doctor": {
                        "type": "string",
                        "description": "The name of the doctor to make the appointment with."
                    }
                },
                "required": ["time", "doctor"],
            },
        },
    }

]

load_dotenv()

doctor_schedule = pd.read_csv("doctors_schedule.csv")


def convert_to_str(time):
    am = True
    if time > 12:
        am = False
        time -= 12
    elif time == 12:
        am = False
    time_str = f'{time} {"am" if am else "pm"}'
    return time_str


def make_appointment(doctor, time):
    booked = doctor_schedule[(doctor_schedule['Doctor'] == doctor) &
                             (doctor_schedule['Time'] == convert_to_str(time))]['Booked'].values[0]
    if booked:
        return "Sorry, that time is already booked. Please choose another time."
    else:
        doctor_schedule.loc[(doctor_schedule['Doctor'] == doctor) &
                            (doctor_schedule['Time'] == convert_to_str(time)), 'Booked'] = True
        doctor_schedule.to_csv("doctors_schedule.csv", index=False)
        return "Appointment booked successfully"


func_utterance = {
    'make_appointment': "Let me see if I can book that time for you. . "}  # the extra period is to trick it into


# speaking


class AppointmentChatbot(BaseChatbot):
    def __init__(self, sys_prompt='', Model='gpt-3.5-turbo'):
        super().__init__()
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.MODEL = Model
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.messages = []
        self.messages.append({"role": "system", "content": sys_prompt})

    def run(self, input_text):
        self.messages.append({"role": "user", "content": input_text})
        finished = False
        while not finished:

            func_call = dict()
            function_call_detected = False

            stream = self.client.chat.completions.create(
                model=self.MODEL,
                messages=self.messages,
                stream=True,
                tools=tools,
                tool_choice="auto",
            )
            for chunk in stream:
                finish_reason = chunk.choices[0].finish_reason
                if chunk.choices[0].delta.tool_calls is not None:
                    function_call_detected = True
                    tool_call = chunk.choices[0].delta.tool_calls[0]
                    if tool_call.function.name:
                        func_call["name"] = tool_call.function.name
                        func_call["id"] = tool_call.id
                        func_call['arguments'] = ''
                        yield func_utterance[func_call['name']]
                    if tool_call.function.arguments:
                        func_call["arguments"] += tool_call.function.arguments
                if function_call_detected and finish_reason == 'tool_calls':
                    self.messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": func_call['id'],
                                "type": "function",
                                "function": {
                                    "name": func_call['name'],
                                    "arguments": func_call['arguments']
                                }
                            }]
                    })
                    # run the function
                    function_response = eval(f"{func_call['name']}(**{func_call['arguments']})")
                    self.messages.append({
                        "tool_call_id": func_call['id'],
                        "role": "tool",
                        "name": func_call['name'],
                        "content": function_response,
                    })
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                if finish_reason == 'stop':
                    finished = True

    def post_process(self, response):
        self.messages.append({"role": "assistant", "content": response})
        return response


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading models... ', device)
    ear = Ear_hf(silence_seconds=2,
                 device=device)

    load_dotenv()

    chatbot = AppointmentChatbot(sys_prompt=prompt)

    mouth = Mouth_xtts(device=device,
                       model_id='tts_models/multilingual/multi-dataset/xtts_v2',
                       speaker='Ana Florence')

    run_chat(mouth, ear, chatbot, verbose=True, stopping_criteria=lambda x: '[END]' in x)
