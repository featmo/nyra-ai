from RealtimeSTT import AudioToTextRecorder

class TranscriptionEngine:
    def __init__(self):
        self.transcription = None

    def process_text(self, text) -> dict[str, str]:
        print(text)
        self.transcription = text
        # return {"transcription": self.transcription}

    def transcribe_mic(self):
        recorder = AudioToTextRecorder()
        while True:
            recorder.text(self.process_text)
        recorder.text(process_text)