from openvoicechat.tts.tts_elevenlabs import Mouth_elevenlabs as Mouth
from openvoicechat.llm.base import BaseChatbot
from openvoicechat.stt.stt_hf import Ear_hf as Ear
from openvoicechat.utils import run_chat
from dotenv import load_dotenv
import os
from openai import OpenAI
from prompts import appointment_prompt as prompt
from prompts import tools
import numpy as np

load_dotenv()

booked = [bool(x) for x in np.random.randint(0, 2, 13)]

def get_available_times():
    times = []
    for hour, is_booked in enumerate(booked, start=9):
        if not is_booked:
            # Format time in 24-hour format and append to list
            formatted_time = f"{hour if hour <= 12 else hour - 12} {'am' if hour < 12 or hour == 24 else 'pm'}"
            times.append(formatted_time)
    return ', '.join(times)
def make_appointment(time):
    print("Making appointment at time:", time)
    time = time - 9
    if booked[time] is True:
        return "The time is already booked"
    booked[time] = True
    return "Appointment booked successfully"


func_utterance = {'make_appointment': "Let me see if I can book that time for you. . "} # the fullstop is to trick it into speaking

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
    device = 'cuda'

    print('loading models... ', device)
    ear = Ear(silence_seconds=2, device=device, )

    load_dotenv()

    chatbot = AppointmentChatbot(sys_prompt=prompt.format(get_available_times()))

    mouth = Mouth()

    run_chat(mouth, ear, chatbot, verbose=True, stopping_criteria=lambda x: '[END]' in x)
