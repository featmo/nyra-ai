import sys
import queue
import threading
import multiprocessing
import re
import time
import torch
import sounddevice as sd
from ollama import chat, Message
from piper import PiperVoice
from RealtimeSTT import AudioToTextRecorder
from pathlib import Path

from Config.Configurations import ASSISTANT_NAME, MODEL, CONTEXT_SIZE, SYSTEM_PROMPT
from Tools.UserTools import *

# ---------- shared on/off switches ----------
stop_event = threading.Event()   # set = "stop talking now"
busy_event = threading.Event()   # set = "a response is in progress"
THINKING_BOOL = True

status_alert = None

# ---------- tools: one dict, the list is built from it ----------
TOOLS = {
    "get_date_time": get_date_time,
    "web_search": web_search,
    "web_fetch": web_fetch,
    "save_to_file": save_to_file,
    "list_directory": list_directory,
    "get_current_directory": get_current_directory,
    "get_file_contents": get_file_contents,
    "get_system_specs": get_system_specs,
    "list_processes": list_processes,
    "screen_shot": screen_shot
}

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# ---------- voice ----------
voice = PiperVoice.load("assets/en_GB-cori-medium.onnx")
audio_out = sd.RawOutputStream(samplerate=voice.config.sample_rate,
                               channels=1, dtype="int16")

def clean_for_speech(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_~#>|]", "", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def speak(text: str):
    text = clean_for_speech(text)
    if not text or stop_event.is_set():           # nothing to say, or already stopped
        return
    audio_out.start()
    for chunk in voice.synthesize(text):
        data = chunk.audio_int16_bytes
        # small slices, so the stop flag is checked about every 0.1 s
        for i in range(0, len(data), 4096):
            if stop_event.is_set():
                audio_out.abort()                 # dump audio already in the soundcard buffer
                return
            audio_out.write(data[i:i + 4096])

tts_queue = queue.Queue()

def worker():
    while True:
        sentence = tts_queue.get()
        try:
            speak(sentence)
        except Exception as e:
            print(f"\nError occurred while speaking: {e}")
        finally:
            tts_queue.task_done()                 # always runs, so join() can't hang

threading.Thread(target=worker, daemon=True).start()

# ---------- tools ----------
def run_tool(call) -> str:
    fn = TOOLS.get(call.function.name)
    if fn is None:
        return f"Tool {call.function.name} not found"
    try:
        result = fn(**call.function.arguments)
    except Exception as e:
        print(f"\n⚠️ {call.function.name} failed: {e}")
        return f"Tool error: {e}"
    return str(result)[:4000]

# ---------- the model ----------
def inference(text: str):
    # Target path setup
    target_path = Path("C:/_Dev/ai/nyra-ai/img.png").resolve()
    
    # Check if the file actually exists on the disk
    if target_path.exists():
        messages.append({
            "role": "user", 
            "content": text, 
            "images": [str(target_path)]  # Fix: Must be a list of strings
        })
    else:
        # Fallback if file isn't found or image_path logic bypassed
        messages.append({
            "role": "user", 
            "content": text
        })

    label_printed = False

    while True:                                   # one trip around = one round with the model
        response_stream = chat(
            model=MODEL,
            messages=messages,
            tools=list(TOOLS.values()),
            stream=True,
            think=THINKING_BOOL,
            options={"num_ctx": CONTEXT_SIZE},
            keep_alive="15m",
        )

        reply, tts_batch, tool_calls = "", "", []  # reset every round

        first_chunk = next(response_stream)
        if first_chunk.message.thinking:
            print("Thinking")

        for chunk in response_stream:

            if stop_event.is_set():               # you started talking: stop reading tokens
                break
            msg = chunk.message
            piece = msg.content or ""

            if msg.tool_calls:
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

        if stop_event.is_set():                   # interrupted: save once, as plain text, and leave
            messages.append({"role": "assistant", "content": reply + " [interrupted]"})
            break

        if tts_batch.strip():
            tts_queue.put(tts_batch.strip())

        messages.append(Message(role="assistant", content=reply,
                                tool_calls=tool_calls or None))

        if not tool_calls:                        # no tool request = final answer
            break

        for call in tool_calls:
            messages.append({
                "role": "tool",
                "content": run_tool(call),
                "tool_name": call.function.name,
            })

        # Look through recent history to find the user message that brought the image in
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user" and "images" in msg:
                del msg["images"]
                break  # Remove it once and stop searching

        # loop again: the model sees the results and writes the answer

    print()

# ---------- one response, run in its own thread ----------
def respond(text: str):
    stop_event.clear()                            # new response: forget the last stop
    try:
        inference(text)
    finally:
        tts_queue.join()                          # wait for speech to finish or be skipped
        busy_event.clear()

def status(message: str):
    sys.stdout.write(f"\r\033[K{message}\n")
    sys.stdout.flush()

def main():
    # start up
    print("-----------------------------------------\n")
    print(f"name: {ASSISTANT_NAME}, model: {MODEL}, ctx: {CONTEXT_SIZE}")
    print("-----------------------------------------\n")
    speak("Starting Systems")

    # status alert
    global status_alert 

    def interrupt():
        if busy_event.is_set():                   # she's mid-response
            print("\n[interrupting current response...]")
            stop_event.set()
            time.sleep(1)

    # 1. Update the callback names inside main()
    def on_vad_detect_start():
        global status_alert
        sys.stdout.write("\r\033[K[VAD: Waiting for speech...]\n")
        sys.stdout.flush()
        status_alert = "Waiting"  # Sets flag cleanly

    # callbacks
    def on_recording_start():
        global status_alert
        sys.stdout.write("\r\033[Klistening...\n")
        sys.stdout.flush()
        interrupt()
        tts_queue.put("listening")
        status_alert = "listening"  # Sets flag instead of speaking

    def on_recording_stop():
        global status_alert
        sys.stdout.write("\r\033[Ktranscribing...\n")
        sys.stdout.flush()
        tts_queue.put("transcribing")
        status_alert = "transcribing"  # Sets flag instead of speaking
        

    recorder = AudioToTextRecorder(
        transcription_engine="faster_whisper",
        model="small.en",
        device="cuda" if torch.cuda.is_available() else "cpu",
        spinner=False,
        compute_type="float16" if torch.cuda.is_available() else "float32",
        allowed_latency_limit=150,
        post_speech_silence_duration=1.5,
        on_vad_detect_start=on_vad_detect_start,
        on_recording_start=on_recording_start,    # was a lambda that never interrupted
        on_recording_stop=on_recording_stop,
        # on_wakeword_detected=interrupt,           # fires as soon as the wake word is heard
        
        # wakeword_backend="oww",
        # wake_words="computer",
        # wake_words_sensitivity=0.6,
        # wake_word_buffer_duration=0.1,
        # wake_word_timeout=10.0,
    )
    
    response_thread = None

    try:
        while True:
             
            usertext = recorder.text().strip()    # the mic stays live while she talks
            if not usertext:
                continue

            # --- This is your text stop/cancel check ---
            text_clean = usertext.strip().lower()
            if any(phrase in text_clean for phrase in ["stop listening", "cancel", "nevermind"]):
                print(f"\n🛑 Action canceled by user. Returning to standby...")
                continue  # Skips processing, goes right back to waiting for the wake word
            img_path = ""
            if any(phrase in text_clean for phrase in ["take screenshot", "take screen shot"]):
                img_path = screen_shot()
                print("Screenshot taken!")

            # print("transcribing")
            # tts_queue.put("transcribing")

            print(f"\nuser➜ {usertext}")
            print("\tGenerating response...\n")
            tts_queue.put("Generating response")

            # make sure the old response has fully stopped before starting a new one
            if response_thread and response_thread.is_alive():
                stop_event.set()
                response_thread.join()

            busy_event.set()
            response_thread = threading.Thread(target=lambda: respond(usertext), daemon=True)
            response_thread.start()

    
            os.remove("img.png")
            
        

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_event.set()
        recorder.shutdown()
        try:
             os.remove("img.png")
        except:
            print("file not availabe")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()