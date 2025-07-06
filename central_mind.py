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
    human_intervention_needed: bool
    scratchpad: List[dict]

def build_workflow(llm: Callable, checkpointer=None):

    def parse_react_response(agent: str, response: str):
        thought, result = "", ""
        for line in response.splitlines():
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action Result:"):
                result = line.replace("Action Result:", "").strip()
        print(f"[{agent}] Thought: {thought}\n[{agent}] Action Result: {result}")
        return thought, result

    def update_state(state: GraphState, agent: str, response: str, update_fields: dict):
        thought, result = parse_react_response(agent, response)
        new_scratch = {
            "agent": agent,
            "thought": thought,
            "action_result": result
        }
        return {
            **update_fields,
            "scratchpad": state.get("scratchpad", []) + [new_scratch]
        }

    def extract_terms_node(state: GraphState) -> dict:
        prompt = f"""
You are a compliance analyst AI using ReAct framework.
Text: {state['raw_text']}

Think step-by-step about how to extract key regulatory terms, names, and acronyms.
Then output your Thought, Action, and Action Result.

Format:
Thought: ...
Action: extract_terms
Action Result: ...
"""
        response = llm.invoke(prompt).content
        thought, result = parse_react_response("Term Identifier", response)
        return update_state(state, "Term Identifier", response, {
            "regulatory_terms": [t.strip() for t in result.split(',') if t.strip()]
        })

    def retrieve_context_node(state: GraphState) -> dict:
        terms = ", ".join(state.get("regulatory_terms", []))
        prompt = f"""
You are a compliance AI. Based on the key terms '{terms}', provide external context.

Format:
Thought: ...
Action: retrieve_context
Action Result: ...
"""
        response = llm.invoke(prompt).content
        return update_state(state, "Context Retriever", response, {
            "external_context": parse_react_response("Context Retriever", response)[1]
        })

    def extract_obligation_node(state: GraphState) -> dict:
        prompt = f"""
From the following text, extract the main obligation sentence.

Text: {state['raw_text']}

Format:
Thought: ...
Action: extract_obligation
Action Result: ...
"""
        response = llm.invoke(prompt).content
        return update_state(state, "Obligation Extractor", response, {
            "obligation_sentence": parse_react_response("Obligation Extractor", response)[1]
        })

    def classify_theme_node(state: GraphState) -> dict:
        prompt = f"""
Classify the obligation below into a theme. Return JSON with 'theme' and 'reasoning'.
Obligation: {state['obligation_sentence']}

Format:
Thought: ...
Action: classify_theme
Action Result: {{ "theme": ..., "reasoning": ... }}
"""
        response = llm.invoke(prompt).content
        thought, result = parse_react_response("Theme Classifier", response)
        parsed = json.loads(result)
        return update_state(state, "Theme Classifier", response, {
            "policy_theme": parsed.get("theme"),
            "policy_theme_reasoning": parsed.get("reasoning"),
            "human_intervention_needed": True
        })

    def find_party_node(state: GraphState) -> dict:
        prompt = f"""
Who is responsible for this obligation? Return JSON with 'party' and 'reasoning'.
Obligation: {state['obligation_sentence']}

Format:
Thought: ...
Action: find_party
Action Result: {{ "party": ..., "reasoning": ... }}
"""
        response = llm.invoke(prompt).content
        parsed = json.loads(parse_react_response("Party Locator", response)[1])
        return update_state(state, "Party Locator", response, {
            "responsible_party": parsed.get("party"),
            "responsible_party_reasoning": parsed.get("reasoning")
        })

    def identify_division_node(state: GraphState) -> dict:
        prompt = f"""
Identify business divisions affected by the obligation.
Return JSON with 'divisions' and 'reasoning'.

Obligation: {state['obligation_sentence']}

Format:
Thought: ...
Action: identify_division
Action Result: {{ "divisions": ..., "reasoning": ... }}
"""
        response = llm.invoke(prompt).content
        parsed = json.loads(parse_react_response("Division Identifier", response)[1])
        return update_state(state, "Division Identifier", response, {
            "divisional_impact": parsed.get("divisions"),
            "divisional_impact_reasoning": parsed.get("reasoning")
        })

    def score_risk_node(state: GraphState) -> dict:
        prompt = f"""
Assess compliance risk for the obligation and theme.
Return JSON with 'score' and 'reasoning'.

Obligation: {state['obligation_sentence']}
Theme: {state['policy_theme']}

Format:
Thought: ...
Action: score_risk
Action Result: {{ "score": ..., "reasoning": ... }}
"""
        response = llm.invoke(prompt).content
        parsed = json.loads(parse_react_response("Risk Scorer", response)[1])
        return update_state(state, "Risk Scorer", response, {
            "risk_score": parsed.get("score"),
            "risk_score_reasoning": parsed.get("reasoning")
        })

    def summarize_node(state: GraphState) -> dict:
        data = {k: v for k, v in state.items() if k not in ['scratchpad', 'raw_text', 'human_intervention_needed']}
        prompt = f"""
Summarize the following analysis in one paragraph:
{json.dumps(data, indent=2)}

Format:
Thought: ...
Action: summarize
Action Result: ...
"""
        response = llm.invoke(prompt).content
        return update_state(state, "Summarizer", response, {
            "summary": parse_react_response("Summarizer", response)[1]
        })

    def qa_critic_node(state: GraphState) -> dict:
        data = {k: v for k, v in state.items() if k != 'scratchpad'}
        prompt = f"""
You are a QA critic. Review this regulatory analysis for consistency.
Return your critique or say 'Analysis appears consistent and complete.'

Analysis:
{json.dumps(data, indent=2)}

Format:
Thought: ...
Action: qa_review
Action Result: ...
"""
        response = llm.invoke(prompt).content
        return update_state(state, "QA Critic", response, {
            "qa_notes": parse_react_response("QA Critic", response)[1]
        })

    def check_for_human_intervention(state: GraphState):
        return "human_intervention_node" if state.get("human_intervention_needed") else "continue"

    def human_intervention_node(state: GraphState):
        current = state.get("policy_theme")
        print("\nHUMAN CHECKPOINT: Theme classified as:", current)
        override = input("Press ENTER to approve, or enter new theme: ").strip()
        return {
            "policy_theme": override if override else current,
            "human_intervention_needed": False
        }

    workflow = StateGraph(GraphState)
    workflow.add_node("extract_terms", extract_terms_node)
    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("extract_obligation", extract_obligation_node)
    workflow.add_node("classify_theme", classify_theme_node)
    workflow.add_node("human_intervention_node", human_intervention_node)
    workflow.add_node("find_party", find_party_node)
    workflow.add_node("identify_division", identify_division_node)
    workflow.add_node("score_risk", score_risk_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("qa_critic", qa_critic_node)

    workflow.set_entry_point("extract_terms")
    workflow.add_edge("extract_terms", "retrieve_context")
    workflow.add_edge("retrieve_context", "extract_obligation")
    workflow.add_edge("extract_obligation", "classify_theme")
    workflow.add_conditional_edges("classify_theme", check_for_human_intervention, {
        "human_intervention_node": "human_intervention_node",
        "continue": "find_party"
    })
    workflow.add_edge("human_intervention_node", "find_party")
    workflow.add_edge("find_party", "identify_division")
    workflow.add_edge("identify_division", "score_risk")
    workflow.add_edge("score_risk", "summarize")
    workflow.add_edge("summarize", "qa_critic")
    workflow.add_edge("qa_critic", END)

    return workflow.compile(checkpointer=checkpointer)
