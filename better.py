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
    async def parse_react_response(agent: str, response: str):
        thought, result = "", ""
        for line in response.splitlines():
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action Result:"):
                result = line.replace("Action Result:", "").strip()
        print(f"[{agent}] Thought: {thought}\n[{agent}] Action Result: {result}")
        return thought, result

    async def update_state(state: GraphState, agent: str, response: str, update_fields: dict):
        thought, result = await parse_react_response(agent, response)
        new_scratch = {
            "agent": agent,
            "thought": thought,
            "action_result": result
        }
        return {
            **update_fields,
            "scratchpad": state.get("scratchpad", []) + [new_scratch]
        }

    async def extract_terms_node(state: GraphState) -> dict:
        agent = "Term Identifier"
        
        # Step 1: Retrieve text from indexed corpus (e.g., via vector search or document store)
        query = state.get("raw_text")  # using raw_text as query input
        retrieved_chunk = await doc_index.search(query)  # <- async retrieval tool
        relevant_text = retrieved_chunk['content']
        
        # Step 2: Ask LLM to extract key terms from that content
        prompt = f"""
    You are a compliance analyst AI using ReAct framework.
    Text: {relevant_text}
    
    Think step-by-step about how to extract key regulatory terms, names, and acronyms.
    Then output your Thought, Action, and Action Result.
    
    Format:
    Thought: ...
    Action: extract_terms
    Action Result: ...
    """
        response = await llm.ainvoke(prompt)
        thought, result = parse_react_response(agent, response.content)
        
        return update_state(state, agent, response.content, {
            "regulatory_terms": [t.strip() for t in result.split(',') if t.strip()]
        })

    async def retrieve_context_node(state: GraphState):
        terms = ", ".join(state.get("regulatory_terms", []))
        prompt = f"""You are a compliance AI. Based on the key terms '{terms}', provide external context.
Format:
Thought: ...
Action: retrieve_context
Action Result: ..."""
        response = (await llm.invoke(prompt)).content
        _, result = await parse_react_response("Context Retriever", response)
        return await update_state(state, "Context Retriever", response, {
            "external_context": result
        })

    async def extract_obligation_node(state: GraphState):
        prompt = f"""Extract the main obligation sentence.
Text: {state['raw_text']}
Format:
Thought: ...
Action: extract_obligation
Action Result: ..."""
        response = (await llm.invoke(prompt)).content
        _, result = await parse_react_response("Obligation Extractor", response)
        return await update_state(state, "Obligation Extractor", response, {
            "obligation_sentence": result
        })

    async def classify_theme_node(state: GraphState):
        prompt = f"""Classify the obligation below into a theme. Return JSON with 'theme' and 'reasoning'.
Obligation: {state['obligation_sentence']}
Format:
Thought: ...
Action: classify_theme
Action Result: {{ "theme": ..., "reasoning": ... }}"""
        response = (await llm.invoke(prompt)).content
        _, result = await parse_react_response("Theme Classifier", response)
        parsed = json.loads(result)
        return await update_state(state, "Theme Classifier", response, {
            "policy_theme": parsed.get("theme"),
            "policy_theme_reasoning": parsed.get("reasoning"),
            "human_intervention_needed": True
        })

    async def find_party_node(state: GraphState):
        prompt = f"""Who is responsible for this obligation? Return JSON with 'party' and 'reasoning'.
Obligation: {state['obligation_sentence']}
Format:
Thought: ...
Action: find_party
Action Result: {{ "party": ..., "reasoning": ... }}"""
        response = (await llm.invoke(prompt)).content
        _, result = await parse_react_response("Party Locator", response)
        parsed = json.loads(result)
        return await update_state(state, "Party Locator", response, {
            "responsible_party": parsed.get("party"),
            "responsible_party_reasoning": parsed.get("reasoning")
        })

    async def identify_division_node(state: GraphState):
        prompt = f"""Identify business divisions affected by the obligation.
Obligation: {state['obligation_sentence']}
Format:
Thought: ...
Action: identify_division
Action Result: {{ "divisions": ..., "reasoning": ... }}"""
        response = (await llm.invoke(prompt)).content
        _, result = await parse_react_response("Division Identifier", response)
        parsed = json.loads(result)
        return await update_state(state, "Division Identifier", response, {
            "divisional_impact": parsed.get("divisions"),
            "divisional_impact_reasoning": parsed.get("reasoning")
        })

    async def score_risk_node(state: GraphState):
        prompt = f"""Assess compliance risk. Return JSON with 'score' and 'reasoning'.
Obligation: {state['obligation_sentence']}
Theme: {state['policy_theme']}
Format:
Thought: ...
Action: score_risk
Action Result: {{ "score": ..., "reasoning": ... }}"""
        response = (await llm.invoke(prompt)).content
        _, result = await parse_react_response("Risk Scorer", response)
        parsed = json.loads(result)
        return await update_state(state, "Risk Scorer", response, {
            "risk_score": parsed.get("score"),
            "risk_score_reasoning": parsed.get("reasoning")
        })

    async def summarize_node(state: GraphState):
        data = {k: v for k, v in state.items() if k not in ['scratchpad', 'raw_text', 'human_intervention_needed']}
        prompt = f"""Summarize this analysis:
{json.dumps(data, indent=2)}
Format:
Thought: ...
Action: summarize
Action Result: ..."""
        response = (await llm.invoke(prompt)).content
        _, result = await parse_react_response("Summarizer", response)
        return await update_state(state, "Summarizer", response, {
            "summary": result
        })

    async def qa_critic_node(state: GraphState):
        data = {k: v for k, v in state.items() if k != "scratchpad"}
        prompt = f"""You are a QA critic. Review the analysis for consistency.
Analysis:
{json.dumps(data, indent=2)}
Format:
Thought: ...
Action: qa_review
Action Result: ..."""
        response = (await llm.invoke(prompt)).content
        _, result = await parse_react_response("QA Critic", response)
        return await update_state(state, "QA Critic", response, {
            "qa_notes": result
        })

    def check_for_human_intervention(state: GraphState):
        return "human_intervention_node" if state.get("human_intervention_needed") else "continue"

    async def human_intervention_node(state: GraphState):
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
