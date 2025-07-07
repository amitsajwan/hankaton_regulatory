import os
import json
from typing import TypedDict, List, Callable
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    raw_text: str
    regulatory_terms: List[str]
    external_context: str
    obligation_sentence: str
    policy_theme: str
    policy_theme_reasoning: str
    responsible_party: str
    responsible_party_reasoning: str
    divisional_impact: str
    divisional_impact_reasoning: str
    risk_score: str
    risk_score_reasoning: str
    summary: str
    qa_notes: str
    user_query: str
    user_query_result: str
    human_intervention_needed: bool
    scratchpad: List[dict]
    thoughts: List[dict]


def build_workflow(llm: Callable, checkpointer=None):
    def call_llm(prompt: str):
        response = llm.invoke(prompt).content
        try:
            parsed = json.loads(response)
            return parsed.get("thought", ""), parsed.get("result", "")
        except Exception:
            return "", response

    def log(agent: str, thought: str, result: str):
        print(f"[{agent}]\nThought: {thought}\nResult: {result}\n")

    def update_state(state: GraphState, agent: str, thought: str, result: str, updates: dict):
        log(agent, thought, result)
        state["thoughts"] = state.get("thoughts", []) + [{"agent": agent, "thought": thought}]
        state["scratchpad"] = state.get("scratchpad", []) + [{"agent": agent, "thought": thought, "action_result": result}]
        return updates

    def extract_terms(state: GraphState):
        prompt = f"""
        You are a regulatory parser. Extract key regulatory terms from this text.
        Return as JSON: {{"thought": ..., "result": "term1, term2"}}
        Text: {state['raw_text']}
        """
        thought, result = call_llm(prompt)
        return update_state(state, "Term Identifier", thought, result, {"regulatory_terms": [t.strip() for t in result.split(",")]})

    def retrieve_context(state: GraphState):
        terms = ", ".join(state.get("regulatory_terms", []))
        prompt = f"""
        Based on these terms: {terms}, return contextual explanation.
        Return JSON: {{"thought": ..., "result": ...}}
        """
        thought, result = call_llm(prompt)
        return update_state(state, "Context Retriever", thought, result, {"external_context": result})

    def extract_obligation(state: GraphState):
        prompt = f"""
        Extract core obligation sentence from:
        {state['raw_text']}
        Return JSON: {{"thought": ..., "result": ...}}
        """
        thought, result = call_llm(prompt)
        return update_state(state, "Obligation Extractor", thought, result, {"obligation_sentence": result})

    def classify_theme(state: GraphState):
        prompt = f"""
        Classify this obligation into a theme and provide reasoning.
        Obligation: {state['obligation_sentence']}
        Return JSON: {{"thought": ..., "result": {{"theme": ..., "reasoning": ...}}}}
        """
        thought, result = call_llm(prompt)
        theme, reasoning = result.get("theme"), result.get("reasoning")
        return update_state(state, "Theme Classifier", thought, json.dumps(result), {"policy_theme": theme, "policy_theme_reasoning": reasoning, "human_intervention_needed": True})

    def find_party(state: GraphState):
        prompt = f"""
        Who is responsible for this obligation? Provide party and reasoning.
        Obligation: {state['obligation_sentence']}
        Return JSON: {{"thought": ..., "result": {{"party": ..., "reasoning": ...}}}}
        """
        thought, result = call_llm(prompt)
        return update_state(state, "Party Locator", thought, json.dumps(result), {"responsible_party": result.get("party"), "responsible_party_reasoning": result.get("reasoning")})

    def division_impact(state: GraphState):
        prompt = f"""
        Which business divisions are impacted by this?
        Obligation: {state['obligation_sentence']}
        Return JSON: {{"thought": ..., "result": {{"divisions": ..., "reasoning": ...}}}}
        """
        thought, result = call_llm(prompt)
        return update_state(state, "Division Identifier", thought, json.dumps(result), {"divisional_impact": result.get("divisions"), "divisional_impact_reasoning": result.get("reasoning")})

    def risk_score(state: GraphState):
        prompt = f"""
        What is the risk score (High/Medium/Low) and why?
        Obligation: {state['obligation_sentence']}
        Theme: {state['policy_theme']}
        Return JSON: {{"thought": ..., "result": {{"score": ..., "reasoning": ...}}}}
        """
        thought, result = call_llm(prompt)
        return update_state(state, "Risk Scorer", thought, json.dumps(result), {"risk_score": result.get("score"), "risk_score_reasoning": result.get("reasoning")})

    def summarize(state: GraphState):
        prompt = f"""
        Provide a final summary in 1-2 lines.
        Return JSON: {{"thought": ..., "result": ...}}
        """
        thought, result = call_llm(prompt)
        return update_state(state, "Summarizer", thought, result, {"summary": result})

    def qa_critic(state: GraphState):
        data = {k: v for k, v in state.items() if k not in ["scratchpad", "thoughts"]}
        prompt = f"""
        Review this regulatory analysis for errors or inconsistencies.
        Return JSON: {{"thought": ..., "result": ...}}
        Data: {json.dumps(data)}
        """
        thought, result = call_llm(prompt)
        return update_state(state, "QA Critic", thought, result, {"qa_notes": result})

    def user_query_tool(state: GraphState):
        prompt = f"""
        You are reviewing the current regulatory analysis state.
        A user has asked: "{state.get('user_query')}"
        Think through the data and either update the state or respond.
        Return JSON: {{"thought": ..., "result": ...}}
        State: {json.dumps({k: v for k, v in state.items() if k not in ['scratchpad', 'thoughts']})}
        """
        thought, result = call_llm(prompt)
        return update_state(state, "User Intent Handler", thought, result, {"user_query_result": result})

    def check_for_human(state):
        return "human_intervention" if state.get("human_intervention_needed") else "continue"

    def human_intervention(state):
        print(f"\nHuman check: LLM classified theme as {state['policy_theme']}")
        new_theme = input("Override theme (or ENTER to approve): ").strip()
        if new_theme:
            return {"policy_theme": new_theme, "human_intervention_needed": False}
        return {"human_intervention_needed": False}

    g = StateGraph(GraphState, checkpointer=checkpointer)
    g.add_node("extract_terms", extract_terms)
    g.add_node("retrieve_context", retrieve_context)
    g.add_node("extract_obligation", extract_obligation)
    g.add_node("classify_theme", classify_theme)
    g.add_node("human_intervention", human_intervention)
    g.add_node("find_party", find_party)
    g.add_node("division_impact", division_impact)
    g.add_node("risk_score", risk_score)
    g.add_node("summarize", summarize)
    g.add_node("qa_critic", qa_critic)
    g.add_node("user_query_tool", user_query_tool)

    g.set_entry_point("extract_terms")
    g.add_edge("extract_terms", "retrieve_context")
    g.add_edge("retrieve_context", "extract_obligation")
    g.add_edge("extract_obligation", "classify_theme")
    g.add_conditional_edges("classify_theme", check_for_human, {"human_intervention": "human_intervention", "continue": "find_party"})
    g.add_edge("human_intervention", "find_party")
    g.add_edge("find_party", "division_impact")
    g.add_edge("division_impact", "risk_score")
    g.add_edge("risk_score", "summarize")
    g.add_edge("summarize", "qa_critic")
    g.add_edge("qa_critic", "user_query_tool")
    g.add_edge("user_query_tool", END)

    return g.compile()
