import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agenticpay.openenv_adapter.procure_env_extended import ProcureEnvExtended


app = FastAPI(
    title="ProcureRL Environment",
    description="Multi-agent procurement negotiation RL environment. Meta OpenEnv Hackathon 2026.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: Dict[str, ProcureEnvExtended] = {}


class ResetRequest(BaseModel):
    difficulty: str = "easy"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    session_id: str
    action: str


class SessionResponse(BaseModel):
    session_id: str
    observation: Dict[str, Any]
    info: Dict[str, Any]


class StepResponse(BaseModel):
    session_id: str
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


@app.get("/")
def root():
    return {
        "name": "ProcureRL",
        "version": "0.1.0",
        "description": "Procurement negotiation RL environment",
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/favicon.ico")
def favicon():
    return ""


@app.post("/reset", response_model=SessionResponse)
def reset(request: ResetRequest):
    session_id = str(uuid.uuid4())
    env = ProcureEnvExtended(difficulty=request.difficulty)
    obs, info = env.reset(seed=request.seed)
    _sessions[session_id] = env

    if len(_sessions) > 100:
        oldest = list(_sessions.keys())[0]
        del _sessions[oldest]

    return SessionResponse(session_id=session_id, observation=obs, info=info)


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    env = _sessions.get(request.session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {request.session_id} not found. Call /reset first.",
        )

    try:
        obs, reward, terminated, truncated, info = env.step(request.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return StepResponse(
        session_id=request.session_id,
        observation=obs,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )


@app.get("/state/{session_id}")
def state(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return env.state()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
