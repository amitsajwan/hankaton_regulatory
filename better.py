import os import json from typing import TypedDict, List, Callable from langgraph.graph import StateGraph, END

class GraphState(TypedDict): raw_text: str regulatory_terms: List[str] external_context: str obligation_sentence: str policy_theme: str policy_theme_reasoning: str responsible_party: str responsible_party_reasoning: str divisional_impact: str divisional_impact_reasoning: str risk_score: str risk_score_reasoning: str summary: str qa_notes: str human_intervention_needed: bool screen_output: dict thoughts: List[dict] scratchpad: dict

def build_workflow(llm: Callable, checkpointer=None, websocket=None): async def send_ws(agent: str, msg: str): if websocket: await websocket.send_json({"agent": agent, "message": msg})

async def capture(agent: str, prompt: str):
    await send_ws(agent, f"Prompt: {prompt}")
    result = await llm.invoke(prompt)
    try:
        data = json.loads(result.content)
        await send_ws(agent, f"Thought: {data.get('thought')}")
        await send_ws(agent, f"Result: {data.get('result')}")
        return data
    except Exception as e:
        await send_ws(agent, f"Failed to parse: {result.content}")
        raise e

async def extract_terms(state: GraphState):
    agent = "TermIdentifier"
    prompt = f"Extract key regulatory terms from: {state['raw_text']}. Return JSON with thought and result (list)."
    data = await capture(agent, prompt)
    return {"regulatory_terms": data.get("result", []), "thoughts": state.get("thoughts", []) + [{"agent": agent, "thought": data.get("thought")}]} 

async def retrieve_context(state: GraphState):
    agent = "ContextRetriever"
    prompt = f"Retrieve external context for: {state['regulatory_terms']}. Return JSON with thought and result."
    data = await capture(agent, prompt)
    return {"external_context": data.get("result"), "thoughts": state.get("thoughts", []) + [{"agent": agent, "thought": data.get("thought")}]} 

async def extract_obligation(state: GraphState):
    agent = "ObligationExtractor"
    prompt = f"Extract obligation from text: {state['raw_text']}. Return JSON with thought and result."
    data = await capture(agent, prompt)
    return {"obligation_sentence": data.get("result"), "thoughts": state.get("thoughts", []) + [{"agent": agent, "thought": data.get("thought")}]} 

async def classify_theme(state: GraphState):
    agent = "ThemeClassifier"
    prompt = f"Classify the theme of obligation '{state['obligation_sentence']}' into policy theme. Return JSON with theme, reasoning, thought."
    data = await capture(agent, prompt)
    return {
        "policy_theme": data.get("theme"),
        "policy_theme_reasoning": data.get("reasoning"),
        "human_intervention_needed": True,
        "thoughts": state.get("thoughts", []) + [{"agent": agent, "thought": data.get("thought")}]
    }

async def find_party(state: GraphState):
    agent = "PartyLocator"
    prompt = f"Find responsible party for obligation: {state['obligation_sentence']}. Return JSON with party, reasoning, thought."
    data = await capture(agent, prompt)
    return {
        "responsible_party": data.get("party"),
        "responsible_party_reasoning": data.get("reasoning"),
        "thoughts": state.get("thoughts", []) + [{"agent": agent, "thought": data.get("thought")}]
    }

async def division_impact(state: GraphState):
    agent = "DivisionImpact"
    prompt = f"Identify divisions affected by obligation: {state['obligation_sentence']}. Return JSON with divisions, reasoning, thought."
    data = await capture(agent, prompt)
    return {
        "divisional_impact": data.get("divisions"),
        "divisional_impact_reasoning": data.get("reasoning"),
        "thoughts": state.get("thoughts", []) + [{"agent": agent, "thought": data.get("thought")}]
    }

async def risk_scorer(state: GraphState):
    agent = "RiskScorer"
    prompt = f"Assess risk for obligation '{state['obligation_sentence']}' under theme '{state['policy_theme']}'. Return JSON with score, reasoning, thought."
    data = await capture(agent, prompt)
    return {
        "risk_score": data.get("score"),
        "risk_score_reasoning": data.get("reasoning"),
        "thoughts": state.get("thoughts", []) + [{"agent": agent, "thought": data.get("thought")}]
    }

async def summarize(state: GraphState):
    agent = "Summarizer"
    context = json.dumps({k: v for k, v in state.items() if k != "thoughts"})
    prompt = f"Summarize the following regulatory analysis:

{context}. Return JSON with thought and result." data = await capture(agent, prompt) return { "summary": data.get("result"), "thoughts": state.get("thoughts", []) + [{"agent": agent, "thought": data.get("thought")}] }

async def qa_critic(state: GraphState):
    agent = "QACritic"
    prompt = f"Perform QA on regulatory analysis:

{json.dumps(state)}. Return JSON with thought and result." data = await capture(agent, prompt) return { "qa_notes": data.get("result"), "thoughts": state.get("thoughts", []) + [{"agent": agent, "thought": data.get("thought")}] }

async def user_query_handler(state: GraphState):
    agent = "UserIntentHandler"
    prompt = f"A user raised a question or concern: '{state.get('user_query', '')}'. Based on this, update the analysis state if needed. Return JSON with any updates and thought."
    data = await capture(agent, prompt)
    updates = data.get("result", {})
    if isinstance(updates, dict):
        return {**updates, "thoughts": state.get("thoughts", []) + [{"agent": agent, "thought": data.get("thought")}]} 
    return {}

def check_for_human(state: GraphState):
    return "human" if state.get("human_intervention_needed") else "next"

def human(state: GraphState):
    new_theme = input(f"Theme was: {state['policy_theme']}. Enter override or press Enter to approve:").strip()
    return {"policy_theme": new_theme or state["policy_theme"], "human_intervention_needed": False}

workflow = StateGraph(GraphState, checkpointer=checkpointer)
workflow.add_node("terms", extract_terms)
workflow.add_node("context", retrieve_context)
workflow.add_node("obligation", extract_obligation)
workflow.add_node("theme", classify_theme)
workflow.add_node("human", human)
workflow.add_node("party", find_party)
workflow.add_node("division", division_impact)
workflow.add_node("risk", risk_scorer)
workflow.add_node("summary", summarize)
workflow.add_node("qa", qa_critic)
workflow.add_node("user_query", user_query_handler)

workflow.set_entry_point("terms")
workflow.add_edge("terms", "context")
workflow.add_edge("context", "obligation")
workflow.add_edge("obligation", "theme")
workflow.add_conditional_edges("theme", check_for_human, {"human": "human", "next": "party"})
workflow.add_edge("human", "party")
workflow.add_edge("party", "division")
workflow.add_edge("division", "risk")
workflow.add_edge("risk", "summary")
workflow.add_edge("summary", "qa")
workflow.add_edge("qa", END)

return workflow.compile()
