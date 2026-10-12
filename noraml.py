import sys
import queue
import threading
import time
from ollama import chat, Message, web_search, web_fetch
from RealtimeSTT import AudioToTextRecorder
import sounddevice as sd
from piper import PiperVoice
from Config.Configurations import ASSISTANT_NAME, MODEL, CONTEXT_SIZE, SYSTEM_PROMPT
from Tools.UserTools import get_date_time
import torch

THINKING_BOOL = False  # global variable to track thinking mode
def set_thinking_mode(mode: bool):
    """Sets the thinking mode for the model."""
    global THINKING_BOOL
    THINKING_BOOL = mode

TOOLS = {"get_date_time": get_date_time, 'web_search': web_search, 'web_fetch': web_fetch, 
         'set_thinking_mode': set_thinking_mode}

messages = [
    {"role": "system", 
     "content": SYSTEM_PROMPT}
] 

voice = PiperVoice.load("en_GB-cori-high.onnx")
stream = sd.RawOutputStream(samplerate=voice.config.sample_rate,
                                channels=1, dtype="int16")

# threading and queue for TTS to avoid blocking the main thread
tts_queue = queue.Queue()

def worker():
    while True:
        sentence = tts_queue.get()
        speak(sentence)
        tts_queue.task_done()

threading.Thread(target=worker, daemon=True).start()

def speak(text: str):
    stream.start()
    for chunk in voice.synthesize(text):
        stream.write(chunk.audio_int16_bytes)

def inference(text: str):

    messages.append({"role": "user", "content": text})

    label_printed = False

    while True:                                   # one trip around = one round with the model
        stream = chat(
            model=MODEL,
            messages=messages,
            tools=[get_date_time, web_search, web_fetch, set_thinking_mode],
            stream=True,
            think=THINKING_BOOL,
            options={"num_ctx": CONTEXT_SIZE},
            keep_alive="15m",
        )

        reply, tts_batch, tool_calls = "", "", []  # reset every round

        for chunk in stream:
            msg = chunk.message
            piece = msg.content or ""

            if msg.tool_calls:                     # the model is asking for a tool
                tool_calls.extend(msg.tool_calls)

            if piece:
                if not label_printed:
                    print("nyra➜ ", end="", flush=True)
                    label_printed = True
                print(piece, end="", flush=True)
                reply += piece
                tts_batch += piece

                if piece.rstrip().endswith((".", "!", "?")):
                    tts_queue.put(tts_batch.strip())
                    tts_batch = ""

        if tts_batch.strip():
            tts_queue.put(tts_batch.strip())

        # Record what the model did, so the next round can see it asked for a tool
        messages.append(Message(role="assistant", content=reply,
                                tool_calls=tool_calls or None))

        if not tool_calls:                         # no request = final answer, stop looping
            break

        for call in tool_calls:                    # the model asked, we run it
            # print(f"\n🔧 {call.function.name}({call.function.arguments})")
            result = TOOLS[call.function.name](**call.function.arguments)
            messages.append({
                "role": "tool",
                "content": str(result),
                "tool_name": call.function.name,
            })
        # loop again: the model is called with the result and writes the answer

    print()

def status(message: str):
    # new line before and after the message for better visibility
    sys.stdout.write(f"\r\033[K{message}\n")
    sys.stdout.flush()

def main():
    recorder = AudioToTextRecorder(
        model="tiny.en",
        device="cuda" if torch.cuda.is_available() else "cpu",
        spinner=False,
        compute_type="float32"
        allowed_latency_limit=150,
        on_vad_detect_start=lambda: status("waiting___"),
        on_recording_start=lambda: status("listening___"),
        on_recording_stop=lambda: status("transcribing___"),
    )
    print(f"nyra-ai: {ASSISTANT_NAME}, {MODEL}, {CONTEXT_SIZE}")
    print("------------------------------------------------\n")
    try:
        while True:
            # add mic emoji to indicate listening
            usertext = recorder.text().strip()  # Get the transcribed text from the microphone
            if usertext:
                status(f"\nuser➜ {usertext}")
                status("⏳ Generating response...")
                inference(usertext)
                tts_queue.join()  # Wait for the TTS queue to be processed
                print("\n")
            recorder.clear_audio_queue()

    except KeyboardInterrupt:
        pass

    finally:
        recorder.shutdown()



if __name__ == "__main__":
    main()


    