from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google import genai
import speech_recognition as sr
import tempfile, shutil, os

# ---- Google GenAI client (uses your existing GEMINI_API_KEY) ----
client = genai.Client()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)

# Language mapping for STT
LANG_MAP = {"en": "en-IN", "hi": "hi-IN", "mr": "mr-IN"}

class PromptRequest(BaseModel):
    prompt: str
    language: str = "en"     # "en" | "hi" | "mr"

@app.post("/generate")
def generate(req: PromptRequest):
    query = (
        f"Give small farmers simple guidance in {req.language}. "
        f"Start with a brief summary, then a detailed report. "
        f"Include loans, tools, transport, markets, daily prices. "
        f"Prompt: {req.prompt}"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
    )
    return {"result": response.text}

@app.post("/speech-to-generate")
async def speech_to_generate(
    language: str = Form("en"),      # form field
    file: UploadFile = File(...)     # multipart file
):
    # 1) Require WAV (simplest + most reliable)
    if not file.filename.lower().endswith((".wav", ".wave")):
        raise HTTPException(status_code=400, detail="Please upload a WAV file (PCM).")

    # 2) Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # 3) Speech-to-text
    try:
        recog = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio = recog.record(source)
        stt_text = recog.recognize_google(audio, language=LANG_MAP.get(language, "en-IN"))
    except sr.UnknownValueError:
        raise HTTPException(status_code=422, detail="Could not understand audio.")
    except sr.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Speech service error: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # 4) Send transcript to Gemini
    query = (
        f"Give small farmers simple guidance in {language}. "
        f"Start with a brief summary, then a detailed report. "
        f"Include loans, tools, transport, markets, daily prices. "
        f"User said: {stt_text}"
    )
    ai = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
    )
    return {"transcript": stt_text, "result": ai.text}
