from fastapi import FastAPI, UploadFile, File
import whisper
import uuid
import os
import requests

app = FastAPI()
model = whisper.load_model("base")

TRANSLATOR_URL = os.getenv("TRANSLATOR_URL")  # e.g., Railway URL for translator server

@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)):
    local_audio_path = f"/tmp/{uuid.uuid4()}.mp3"
    with open(local_audio_path, "wb") as f:
        f.write(file.file.read())

    result = model.transcribe(local_audio_path)
    transcript_text = result['text']

    # Send transcript to Translator Server
    response = requests.post(
        f"{TRANSLATOR_URL}/translate",
        json={"transcript_text": transcript_text, "target_lang": "fr"}
    )
    return response.json()