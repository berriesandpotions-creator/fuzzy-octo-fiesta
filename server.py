import os
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from fastapi.middleware.cors import CORSMiddleware

api_key = os.getenv("Gemini_API_Key")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set in environment!")

client = genai.Client(api_key=api_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str
    language: str = "en"

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

def test_basic():
    assert 1 + 1 == 2
