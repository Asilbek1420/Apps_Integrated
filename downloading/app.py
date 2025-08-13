from fastapi import FastAPI
from pydantic import BaseModel
from yt_dlp import YoutubeDL
import uuid
import os
import requests

app = FastAPI()

TRANSCRIBER_URL = os.getenv("TRANSCRIBER_URL")  # e.g., Railway URL for transcriber server

class DownloadRequest(BaseModel):
    url: str

@app.post("/download")
def download(req: DownloadRequest):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '/tmp/%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(req.url, download=True)
        audio_path = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')

    # Send audio to Transcriber Server
    files = {'file': open(audio_path, 'rb')}
    response = requests.post(f"{TRANSCRIBER_URL}/transcribe", files=files)
    return response.json()
