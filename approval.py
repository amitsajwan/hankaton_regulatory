from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langgraph.graph import StateGraph, END
import asyncio

app = FastAPI()

# --- Define Graph State ---
from typing import TypedDict

class State(TypedDict):
    message: str
    approved: bool

# --- Define LangGraph Nodes ---
async def generate_message_node(state: State) -> State:
    return {
        **state,
        "message": "This is a generated message. Approve?",
        "approved": False
    }

async def wait_for_user_node(state: State, websocket: WebSocket) -> State:
    await websocket.send_json({"type": "user_intervention", "payload": state})
    user_input = await websocket.receive_json()
    approved = user_input.get("approved", False)
    return {
        **state,
        "approved": approved
    }

# --- Conditional Routing ---
async def check_approval(state: State):
    return "end" if state["approved"] else "generate"

# --- LangGraph Workflow ---
def build_graph(websocket: WebSocket):
    workflow = StateGraph(State)
    workflow.add_node("generate", generate_message_node)
    workflow.add_node("wait", lambda state: wait_for_user_node(state, websocket))
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "wait")
    workflow.add_conditional_edges("wait", check_approval, {
        "generate": "generate",
        "end": END
    })
    return workflow.compile()

# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    graph = build_graph(websocket)
    state = {"message": "", "approved": False}
    async for s in graph.stream(state):
        pass  # Output is handled in wait_for_user_node
    await websocket.send_json({"type": "complete", "payload": "Workflow complete."})
