import os
import json
from typing import TypedDict, List, Callable
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    raw_text: str
    markdown_text: str
    regulatory_terms: List[str]
    external_context: str
    obligation_sentence: str
    policy_theme: str
    responsible_party: str
    summary: str
    human_intervention_needed: bool
    regulation_summary: str
    compliance_dates: str
    ownership: str
    divisional_impact: str
    risk_score: str
    qa_notes: str
    screen_output: dict
    thoughts: List[dict]
    scratchpad: dict

def build_workflow(llm: Callable, checkpointer=None, websocket=None):

    def invoke_llm(prompt: str) -> str:
        return llm.invoke(prompt)

    def send_intermediate_message(agent: str, msg: str):
        print(f"[{agent}] {msg}")
        if websocket:
            import asyncio
            asyncio.create_task(websocket.send_json({"agent": agent, "message": msg}))

    def add_message(state: GraphState, agent: str, msg: str):
        send_intermediate_message(agent, msg)
        scratchpad = state.get("scratchpad", {})
        scratchpad.setdefault("intermediate_messages", []).append({"agent": agent, "message": msg})
        state["thoughts"] = state.get("thoughts", []) + [{"agent": agent, "thought": msg}]
        return scratchpad

    tool_map = {}

    def register_tool(name: str):
        def decorator(func):
            tool_map[name] = func
            return func
        return decorator

    @register_tool("extract_terms")
    def extract_terms(state: GraphState) -> str:
        prompt = f"Thought: Identify key terms.\nAction: extract_terms\nAction Input: {state['raw_text']}"
        return invoke_llm(prompt)

    @register_tool("retrieve_context")
    def retrieve_context(state: GraphState) -> str:
        terms = ", ".join(state.get("regulatory_terms", []))
        prompt = f"Thought: Retrieve external context.\nAction: retrieve_context\nAction Input: {terms}"
        return invoke_llm(prompt)

    @register_tool("extract_obligation")
    def extract_obligation(state: GraphState) -> str:
        prompt = f"Thought: Extract obligation from text.\nAction: extract_obligation\nAction Input: {state['raw_text']}"
        return invoke_llm(prompt)

    @register_tool("classify_theme")
    def classify_theme(state: GraphState) -> str:
        prompt = f"Thought: Classify the theme.\nAction: classify_theme\nAction Input: {state['obligation_sentence']}"
        return invoke_llm(prompt)

    @register_tool("find_party")
    def find_party(state: GraphState) -> str:
        prompt = f"Thought: Find responsible party.\nAction: find_party\nAction Input: {state['obligation_sentence']}"
        return invoke_llm(prompt)

    @register_tool("summarize")
    def summarize(state: GraphState) -> str:
        prompt = f"Thought: Summarize regulation.\nAction: summarize\nAction Input: {state['obligation_sentence']}, {state['policy_theme']}, {state['external_context']}"
        return invoke_llm(prompt)

    @register_tool("division_function_identifier")
    def division_function_identifier(state: GraphState) -> str:
        prompt = f"Thought: Identify relevant divisions or functions.\nAction: division_function_identifier\nAction Input: {state['obligation_sentence']}"
        return invoke_llm(prompt)

    @register_tool("risk_scoring")
    def risk_scoring(state: GraphState) -> str:
        prompt = f"Thought: Score regulatory risk (e.g., High, Medium, Low).\nAction: risk_scoring\nAction Input: {state['obligation_sentence']}, {state['policy_theme']}"
        return invoke_llm(prompt)

    @register_tool("qa_critic")
    def qa_critic(state: GraphState) -> str:
        prompt = f"Thought: Perform QA review.\nAction: qa_critic\nAction Input: {json.dumps(state)}"
        return invoke_llm(prompt)

    def react_controller(state: GraphState) -> GraphState:
        max_steps = 12
        for _ in range(max_steps):
            if state.get("summary") and state.get("divisional_impact") and state.get("risk_score"):
                break
            for name, func in tool_map.items():
                output = func(state)
                add_message(state, name, output)
                if name == "extract_terms":
                    state["regulatory_terms"] = output.split(", ")
                elif name == "retrieve_context":
                    state["external_context"] = output
                elif name == "extract_obligation":
                    state["obligation_sentence"] = output
                elif name == "classify_theme":
                    state["policy_theme"] = output
                elif name == "find_party":
                    state["responsible_party"] = output
                elif name == "summarize":
                    state["summary"] = output
                elif name == "division_function_identifier":
                    state["divisional_impact"] = output
                elif name == "risk_scoring":
                    state["risk_score"] = output
                elif name == "qa_critic":
                    state["qa_notes"] = output

            state["screen_output"] = {
                "obligation": state.get("obligation_sentence"),
                "policy_theme": state.get("policy_theme"),
                "responsible_party": state.get("responsible_party"),
                "external_context": state.get("external_context"),
                "regulation_summary": state.get("summary"),
                "division": state.get("divisional_impact"),
                "risk_score": state.get("risk_score"),
                "qa": state.get("qa_notes")
            }
        return state

    workflow = StateGraph(GraphState, checkpointer=checkpointer)
    workflow.add_node("controller", react_controller)
    workflow.set_entry_point("controller")
    workflow.add_edge("controller", END)

    return workflow.compile()
