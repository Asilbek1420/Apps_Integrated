from fastapi import FastAPI
from pydantic import BaseModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

app = FastAPI()

# Load the small, open-source M2M-100 model
MODEL_NAME = "facebook/m2m100_418M"  # fully free and open-source
tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)

class TranslateRequest(BaseModel):
    transcript_text: str
    source_lang: str = "en"  # source language code
    target_lang: str = "fr"  # target language code

@app.post("/translate")
def translate(req: TranslateRequest):
    # Set source language
    tokenizer.src_lang = req.source_lang

    # Encode input
    encoded = tokenizer(req.transcript_text, return_tensors="pt")

    # Generate translation
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(req.target_lang)
    )

    # Decode translation
    translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return {"translated_text": translated_text}
