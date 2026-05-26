"""FastAPI REST API for Temporal KBQA.

Run with:
    uvicorn src.api:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /ask    { "question": "..." }  →  answer + provenance
    GET  /health
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="Temporal KBQA API",
    description="Generate-then-Retrieve KBQA with temporal reasoning over Freebase",
    version="1.0.0",
)

_agent = None


class QuestionRequest(BaseModel):
    question: str
    mode: Optional[str] = None  # 'temporal' | 'standard' | None (auto-detect)


class AnswerResponse(BaseModel):
    question: str
    answer: list[str]
    is_temporal: bool
    temporal_signals: list[str]
    sparql_used: Optional[str]
    temporal_constraint: Optional[str]
    reasoning_steps: list[str]
    retries: int


def get_agent():
    """Lazy-initialize the agent (avoid loading LLM at import time)."""
    global _agent
    if _agent is None:
        config_path = os.environ.get('TKBQA_CONFIG', 'configs/inference.yaml')
        if not os.path.exists(config_path):
            raise RuntimeError(
                f"Config file not found: {config_path}. "
                "Set TKBQA_CONFIG env var or create configs/inference.yaml"
            )
        from src.pipeline import TemporalKBQAPipeline
        from src.agent import TemporalQuestionAgent
        pipeline = TemporalKBQAPipeline.from_config(config_path)
        _agent = TemporalQuestionAgent(pipeline)
    return _agent


@app.get('/health')
def health():
    config_path = os.environ.get('TKBQA_CONFIG', 'configs/inference.yaml')
    payload = {
        'status': 'ok',
        'service': 'Temporal KBQA',
        'config_path': config_path,
        'config_exists': os.path.exists(config_path),
    }
    if not payload['config_exists']:
        payload['status'] = 'degraded'
        return payload

    try:
        from src.pipeline import TemporalKBQAPipeline
        pipeline = TemporalKBQAPipeline.from_config(config_path)
        runtime = pipeline.runtime_status()
        payload['runtime'] = runtime
        if (
            not runtime['model_exists']
            or (runtime.get('adapter_only_checkpoint') and not runtime.get('base_model_exists'))
        ):
            payload['status'] = 'degraded'
    except Exception as e:
        payload['status'] = 'degraded'
        payload['runtime_error'] = str(e)
    return payload


@app.post('/ask', response_model=AnswerResponse)
def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")
    if len(request.question) > 2000:
        raise HTTPException(status_code=400, detail="Question too long (max 2000 characters)")

    try:
        agent = get_agent()
        result = agent.run(request.question, mode=request.mode)
        return AnswerResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get('/')
def root():
    return {
        'message': 'Temporal KBQA API',
        'endpoints': {
            'POST /ask': 'Answer a temporal or factual question',
            'GET /health': 'Health check',
        },
        'example': {
            'question': 'Who was the US president before Obama?',
        },
    }
