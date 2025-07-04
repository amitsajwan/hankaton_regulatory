from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from regulatory_analysis_graph import build_workflow
import asyncio

# Dummy LLM for testing – replace with real LLM object
class DummyLLM:
    def invoke(self, prompt: str) -> str:
        return f"[LLM] {prompt[:80]}..."

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/chat")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connected.")

    try:
        while True:
            request = await websocket.receive_json()
            text = request.get("text")

            if not text:
                await websocket.send_json({"agent": "System", "message": "❗ Empty text received."})
                continue

            await websocket.send_json({"agent": "System", "message": "🚀 Running graph..."})

            graph = build_workflow(llm=DummyLLM(), websocket=websocket)

            initial_state = {
                "raw_text": text,
                "scratchpad": {}
            }

            final = None
            async for update in graph.astream(initial_state):
                node = list(update.keys())[0]
                payload = update[node]
                final = payload
                await websocket.send_json({
                    "agent": node,
                    "message": f"✅ Step: {node} complete.",
                    "scratchpad": payload.get("scratchpad", {})
                })

            await websocket.send_json({
                "agent": "Summary",
                "message": final.get("summary", "⚠️ No summary generated.")
            })

    except WebSocketDisconnect:
        print("❌ WebSocket disconnected.")
