import os
import json
from fastapi import FastAPI, WebSocket
from langgraph.graph import StateGraph
from regulatory_graph import build_workflow

class DummyLLM:
    def invoke(self, prompt: str):
        return type("Result", (), {"content": f"Dummy result for: {prompt[:50]}..."})()

app = FastAPI()

@app.websocket("/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        llm = DummyLLM()
        workflow = build_workflow(llm, websocket=websocket)

        input_state = await websocket.receive_json()
        input_state.setdefault("scratchpad", [])

        async for state in workflow.astream(input_state, stream_mode="values"):
            await websocket.send_json({"final_state": {k: v for k, v in state.items() if k != "scratchpad"}})

    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()
