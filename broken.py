import sys
import queue
import threading
import time
import re
from ollama import chat, Message
from RealtimeSTT import AudioToTextRecorder
import sounddevice as sd
from piper import PiperVoice
from Config.Configurations import ASSISTANT_NAME, MODEL, CONTEXT_SIZE, SYSTEM_PROMPT
from Tools.UserTools import get_date_time, web_search, web_fetch

# --- STEP 0: The shared on/off switches ---
stop_event = threading.Event()
busy_event = threading.Event()
THINKING_BOOL = False

def set_thinking_mode(mode: str):
    global THINKING_BOOL
    if "thinking off" in mode.lower(): 
        THINKING_BOOL = False
    if "thinking on" in mode.lower():
        THINKING_BOOL = True
    reply = f"Thinking mode set to {THINKING_BOOL}."
    print(f"\n{reply}")
    tts_queue.put(reply)

TOOLS = {
    "get_date_time": get_date_time, 
    "web_search": web_search, 
    "web_fetch": web_fetch, 
    "set_thinking_mode": set_thinking_mode
}

messages = [{"role": "system", "content": SYSTEM_PROMPT}] 

voice = PiperVoice.load("en_GB-cori-high.onnx")

tts_queue = queue.Queue()

def worker():
    while True:
        sentence = tts_queue.get()
        try:
            if not stop_event.is_set():
                speak(sentence)
        except Exception as e:
            print(f"\nError occurred while speaking: {e}")
        finally:
            tts_queue.task_done()

threading.Thread(target=worker, daemon=True).start()

def clean_for_speech(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_~#>|]", "", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", text)
    return re.sub(r"\s+", " ", text).strip() 

# --- DYNAMIC AUDIO STREAM MANAGEMENT ---
def speak(text: str):
    text = clean_for_speech(text)
    # Create a FRESH stream every time to avoid "device busy" locks
    local_stream = sd.RawOutputStream(
        samplerate=voice.config.sample_rate, 
        channels=1, 
        dtype="int16"
    )
    
    try:
        local_stream.start()
        for chunk in voice.synthesize(text):
            if stop_event.is_set():
                break
            
            data = chunk.audio_int16_bytes
            for i in range(0, len(data), 4096):
                if stop_event.is_set():
                    break
                local_stream.write(data[i:i + 4096])
    except Exception as e:
        print(f"\nError during speech: {e}")
    finally:
        # Always close the local stream to release the OS lock immediately
        try:
            local_stream.stop()
            local_stream.close()
        except Exception:
            pass

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

def inference(text: str):
    messages.append({"role": "user", "content": text})
    label_printed = False
    reply = ""

    while True:
        if stop_event.is_set():
            messages.append({"role": "assistant", "content": reply + " [interrupted]"})
            break

        response_stream = chat(
            model=MODEL,
            messages=messages,
            tools=[get_date_time, web_search, web_fetch, set_thinking_mode],
            stream=True,
            think=THINKING_BOOL,
            options={"num_ctx": CONTEXT_SIZE},
            keep_alive="15m",
        )

        tts_batch, tool_calls = "", []

        for chunk in response_stream:
            if stop_event.is_set():
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

        if tts_batch.strip() and not stop_event.is_set():
            tts_queue.put(tts_batch.strip())

        messages.append(Message(role="assistant", content=reply, tool_calls=tool_calls or None))

        if not tool_calls or stop_event.is_set():
            if stop_event.is_set():
                messages.append({"role": "assistant", "content": reply + " [interrupted]"})
            break

        for call in tool_calls:
            if stop_event.is_set():
                break
            result = run_tool(call)
            messages.append({
                "role": "tool",
                "content": str(result),
                "tool_name": call.function.name,
            })
            
    print()

def respond(text: str):
    stop_event.clear()
    try:
        inference(text)
    finally:
        tts_queue.join()
        busy_event.clear()

def status(message: str):
    sys.stdout.write(f"\r\033[K{message}\n")
    sys.stdout.flush()

def main():
    def on_recording_start():
        status("listening___")
        if busy_event.is_set():
            print("\n[interrupting current response...]")
            stop_event.set()
            # Small buffer to let the audio thread catch the stop signal
            time.sleep(0.05)

    recorder = AudioToTextRecorder(
        model="base.en",
        device="cuda" if torch.cuda.is_available() else "cpu",
        spinner=False,
        on_vad_detect_start=lambda: status("waiting___"),
        on_recording_start=on_recording_start,
        on_recording_stop=lambda: status("transcribing___"),
    )
    
    print(f"system: {ASSISTANT_NAME}, {MODEL}, {CONTEXT_SIZE}")
    print("------------------------------------------------\n")
    
    response_thread = None

    try:
        while True:
            usertext = recorder.text().strip()
            if not usertext:
                continue
                
            status(f"\nuser➜ {usertext}")
            status("⏳ Generating response...")
            
            if response_thread and response_thread.is_alive():
                stop_event.set()
                response_thread.join()
            
            busy_event.set()
            response_thread = threading.Thread(target=respond, args=(usertext,), daemon=True)
            response_thread.start()

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_event.set()
        recorder.shutdown()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()