from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from typing import Literal
import json
import os
import logging

logger = logging.getLogger('greatguru.chat')

app = FastAPI(title="GreatGuru GATE ME API")
origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "https://pradeep7095.github.io").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

TEACHER_RULES = """
You are GreatGuru, an autonomous GATE Mechanical Engineering teacher. Teach in clear English from fundamentals.
Teach ONE concise stage per response (roughly 100-180 words), then either continue or check understanding.
Use a progression: introduction, prerequisites, concept, visualization, derivation, solved example,
understanding check, practice, traps, recap. Never skip units, assumptions or prerequisite explanations.
After at most three explanatory responses, ask exactly one meaningful question and WAIT for the student.
When the student answers, evaluate the response, explain errors kindly, and proceed only if understood.
For CONTINUE, teach the next logical stage. For START CLASS, introduce the requested topic.
For interruptions, address the student's question first and then resume the topic.
Never fabricate official GATE previous-year questions, years, marks or citations; label unverified
questions as original practice questions.
You can request one of the existing visualizations only when genuinely relevant:
otto (four-stroke engine/Otto cycle), slider (slider crank), beam (SFD/BMD),
fluid (Venturi/Bernoulli), heat (conduction). Otherwise choose null.
Return a JSON object with keys:
reply (plain text lesson narration), await_answer (boolean), animation (one of
otto,slider,beam,fluid,heat or null), stage (short string).
Set await_answer=true whenever the lesson ends with a question requiring a student response.
Do not output Markdown code fences.
"""

class HistoryTurn(BaseModel):
    role: Literal["user", "ai"]
    text: str = Field(max_length=4000)

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=3000)
    topic: str = Field(default="Engineering Mathematics → Linear Algebra", max_length=200)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=12)
    # Kept for compatibility with older clients; intentionally ignored.
    system_prompt: str | None = None

@app.get("/")
async def health_check():
    return {"status": "online", "service": "GreatGuru GATE Mechanical API"}

@app.post("/chat")
async def chat(req: ChatRequest):
    if not client:
        raise HTTPException(status_code=503, detail="AI service not configured")
    context = "\n".join(
        ("STUDENT: " if h.role == "user" else "TEACHER: ") + h.text
        for h in req.history[-10:]
    )
    user_content = (
        f"Active topic: {req.topic}\n"
        f"Recent lesson dialogue:\n{context}\n"
        f"Latest student instruction: {req.message}"
    )
    try:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=TEACHER_RULES,
                response_mime_type="application/json",
                temperature=0.35,
            ),
        )
        result = json.loads(response.text or "{}")
        reply = result.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("Empty teacher response")
        animation = result.get("animation")
        if animation not in ("otto", "slider", "beam", "fluid", "heat"):
            animation = None
        return {
            "reply": reply,
            "await_answer": result.get("await_answer") is True,
            "animation": animation,
            "stage": str(result.get("stage", "Lesson"))[:60],
        }
    except (ValueError, json.JSONDecodeError):
        logger.exception("GreatGuru AI returned invalid JSON or an empty reply")
        raise HTTPException(status_code=502, detail="Invalid AI response")
    except Exception as exc:
        # Avoid logging secrets or full prompts; exception class and HTTP status
        # are enough to distinguish authentication, quota, and transport failures.
        error_type = type(exc).__name__
        upstream_status = getattr(exc, "code", None)
        if not isinstance(upstream_status, int):
            upstream_status = getattr(exc, "status_code", None)
        logger.error("GreatGuru Gemini request failed: type=%s upstream_status=%s", error_type, upstream_status)
        raise HTTPException(status_code=502, detail="AI service temporarily unavailable")
