from openvoicechat.tts.tts_xtts import Mouth_xtts
from openvoicechat.llm.base import BaseChatbot
from openvoicechat.stt.stt_hf import Ear_hf
from openvoicechat.utils import run_chat
from dotenv import load_dotenv
import os
from openai import OpenAI
from prompts import appointment_prompt as prompt
from prompts import tools
import torch
import pandas as pd

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
    print("Making appointment at time:", time)
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
