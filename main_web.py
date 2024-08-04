from openvoicechat.tts.tts_xtts import Mouth_xtts
from openvoicechat.llm.llm_gpt import Chatbot_gpt
from openvoicechat.stt.stt_hf import Ear_hf
from openvoicechat.utils import run_chat
from dotenv import load_dotenv
import os
from prompts import appointment_prompt as prompt
from prompts import tools
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import threading
import queue
from openvoicechat.utils import Listener_ws, Player_ws
from main import make_appointment, get_available_times, func_utterance

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print('connected')

    input_queue = queue.Queue()
    output_queue = queue.Queue()
    listener = Listener_ws(input_queue)
    player = Player_ws(output_queue)

    print('loading models... ', device)
    ear = Ear_hf(silence_seconds=1.5,
                 device=device,
                 listener=listener)
    load_dotenv()

    await websocket.send_bytes('none'.encode())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    chatbot = Chatbot_gpt(sys_prompt=prompt, api_key=OPENAI_API_KEY,
                          tool_choice='auto', tools=tools,
                          tool_utterances=func_utterance,
                          functions={'make_appointment': make_appointment,
                                     'get_available_times': get_available_times})

    mouth = Mouth_xtts(device=device,
                       model_id='tts_models/multilingual/multi-dataset/xtts_v2',
                       speaker='Ana Florence',
                       player=player)

    await websocket.send_bytes('none'.encode())
    _ = mouth.run_tts('Hello, I am your assistant. How can I help you today?')  # warm up the model
    ear.transcribe(_)  # warm up the model
    starting_msg = 'Hello, I am Anne from the city hospital, would you like to make an appointment?'
    t = threading.Thread(target=run_chat, args=(mouth, ear, chatbot, True, lambda x: '[END]' in x, starting_msg))
    t.start()
    await websocket.send_bytes('none'.encode())
    try:
        while t.is_alive():
            data = await websocket.receive_bytes()
            if listener.listening:
                input_queue.put(data)
            if not output_queue.empty():
                response_data = output_queue.get_nowait()
            else:
                response_data = 'none'.encode()
            await websocket.send_bytes(response_data)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        del mouth
        del ear
        import torch
        torch.cuda.empty_cache()
    finally:
        await websocket.close()


@app.get("/")
def read_root():
    return FileResponse('static/stream_audio.html')


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000,
                ssl_keyfile="localhost+2-key.pem", ssl_certfile="localhost+2.pem")
