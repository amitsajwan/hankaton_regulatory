from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
import asyncio

# --- Define state ---
class GraphState(TypedDict):
    message: str
    approved: bool

# --- Shared message queue ---
class UserInteractionChannel:
    def __init__(self):
        self.queue = asyncio.Queue()

    async def send(self, data):
        await self.queue.put(data)

    async def receive(self):
        return await self.queue.get()

# --- Nodes ---
async def generate_message_node(state: GraphState) -> dict:
    return {"message": "Should we proceed with publishing the regulatory update?", "approved": False}

async def wait_for_user_node(state: GraphState, channel: UserInteractionChannel) -> dict:
    await channel.send({"type": "intervention_required", "message": state["message"]})
    result = await channel.receive()
    if result.get("approved"):
        return {"approved": True}
    return {"approved": False}  # Loop again

async def should_continue(state: GraphState) -> Literal["proceed", "wait_again"]:
    return "proceed" if state.get("approved") else "wait_again"

async def complete_node(state: GraphState) -> dict:
    return {"message": "Approval received. Proceeding with update."}

# --- LangGraph workflow ---
def build_graph(channel: UserInteractionChannel):
    builder = StateGraph(GraphState)
    builder.add_node("generate", generate_message_node)
    builder.add_node("wait_for_user", lambda state: wait_for_user_node(state, channel))
    builder.add_node("complete", complete_node)

    builder.set_entry_point("generate")
    builder.add_edge("generate", "wait_for_user")
    builder.add_conditional_edges("wait_for_user", should_continue, {
        "proceed": "complete",
        "wait_again": "wait_for_user"
    })
    builder.add_edge("complete", END)
    return builder.compile()

# --- FastAPI server ---
app = FastAPI()
user_channels = {}  # {client_id: UserInteractionChannel}

def get_channel(client_id: str):
    if client_id not in user_channels:
        user_channels[client_id] = UserInteractionChannel()
    return user_channels[client_id]

@app.websocket("/chat/{client_id}")
async def chat(client_ws: WebSocket, client_id: str):
    await client_ws.accept()
    channel = get_channel(client_id)
    graph = build_graph(channel)

    async def stream_handler():
        async for event in graph.stream({"message": "", "approved": False}):
            yield event

    async def receive_loop():
        try:
            while True:
                data = await client_ws.receive_json()
                if data.get("type") == "user_decision":
                    await channel.send(data.get("payload", {}))
        except WebSocketDisconnect:
            pass

    async def stream_loop():
        async for event in stream_handler():
            await client_ws.send_json({"type": "event", "payload": event})

    await asyncio.gather(receive_loop(), stream_loop())
