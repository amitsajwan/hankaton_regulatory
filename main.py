from fastapi import FastAPI, WebSocket
from regulatory_graph import build_workflow, GraphState
from langchain_core.language_models import BaseChatModel
from typing import Dict, Any
import asyncio

app = FastAPI()

# Dummy LLM with `invoke()` method (replace with real one)
class DummyLLM(BaseChatModel):
    def _call(self, messages, **kwargs):
        return "Dummy response"

    @property
    def _llm_type(self) -> str:
        return "dummy"

    def invoke(self, prompt: str) -> str:
        print(f"[LLM RECEIVED PROMPT]\n{prompt}\n")
        return "{\"result\": \"Sample output\", \"reasoning\": \"Mock reasoning from model.\"}"

llm = DummyLLM()
workflow = build_workflow(llm)

@app.websocket("/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()
    data: Dict[str, Any] = await websocket.receive_json()

    if "raw_text" not in data:
        await websocket.send_json({"error": "Missing 'raw_text' in request."})
        return

    initial_state: GraphState = {
        "raw_text": data["raw_text"],
        "regulatory_terms": [],
        "external_context": "",
        "obligation_sentence": "",
        "policy_theme": "",
        "policy_theme_reasoning": "",
        "responsible_party": "",
        "responsible_party_reasoning": "",
        "divisional_impact": "",
        "divisional_impact_reasoning": "",
        "risk_score": "",
        "risk_score_reasoning": "",
        "summary": "",
        "qa_notes": "",
        "human_intervention_needed": False,
        "scratchpad": []
    }

    async for state in workflow.astream(initial_state):
        await websocket.send_json({"state": {k: v for k, v in state.items() if k != "scratchpad"}})

    await websocket.send_json({"done": True})
