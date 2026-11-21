from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import os

app = FastAPI(title="GreatGuru GATE ME API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


class ChatRequest(BaseModel):
    message: str
    system_prompt: str = ""
    topic: str = "Engineering Mathematics"


@app.get("/")
async def health_check():
    return {"status": "online", "service": "GreatGuru GATE Mechanical API"}


@app.post("/chat")
async def chat(req: ChatRequest):
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing in Render Environment Variables.")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=req.message,
            config=types.GenerateContentConfig(
                system_instruction=req.system_prompt,
                temperature=0.4,
            ),
        )
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
