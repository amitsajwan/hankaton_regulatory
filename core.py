# This file defines the graph logic and tool behavior
# Enhanced with additional agents for human-like decisions and expanded extraction

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
        return scratchpad

    def pdf_to_text_extractor(state: GraphState) -> GraphState:
        agent = "PDF Extractor"
        msg = "Reading the uploaded PDF and extracting raw text..."
        scratchpad = add_message(state, agent, msg)
        return {"raw_text": state["raw_text"], "scratchpad": scratchpad}

    def regulatory_description_extractor(state: GraphState) -> GraphState:
        agent = "Regulation Summary Extractor"
        prompt = f"""Extract the following from the regulation text:\n- Regulation Description\n- Extraterritorial impact (Yes/No)\n- Explanation of extraterritorial impact\n- Stage of legislation\n- Regulation Type\nText:\n{state['raw_text']}"""
        result = invoke_llm(prompt)
        scratchpad = add_message(state, agent, f"Extracted Regulation Summary: {result}")
        return {"regulation_summary": result, "scratchpad": scratchpad}

    def text_to_markdown_formatter(state: GraphState) -> GraphState:
        agent = "Markdown Formatter"
        msg = "Formatting the extracted text into a structured markdown..."
        markdown = f"## Regulatory Requirement\n\n{state['raw_text']}"
        scratchpad = add_message(state, agent, msg)
        return {"markdown_text": markdown, "scratchpad": scratchpad}

    def regulatory_term_identifier(state: GraphState) -> GraphState:
        agent = "Term Identifier"
        prompt = f"Identify key regulatory terms, dates, and codes in the text:\n\n{state['raw_text']}"
        terms = invoke_llm(prompt)
        scratchpad = add_message(state, agent, f"Identified terms: {terms}")
        return {"regulatory_terms": terms.split(", "), "scratchpad": scratchpad}

    def external_knowledge_retriever(state: GraphState) -> GraphState:
        agent = "Knowledge Retriever"
        query = ", ".join(state["regulatory_terms"])
        prompt = f"Provide contextual background for the following terms:\n{query}"
        context = invoke_llm(prompt)
        scratchpad = add_message(state, agent, f"Retrieved context: {context}")
        return {"external_context": context, "scratchpad": scratchpad}

    def obligation_sentence_extractor(state: GraphState) -> GraphState:
        agent = "Obligation Extractor"
        prompt = f"Extract obligation clauses with 'must', 'shall', 'required' from:\n{state['raw_text']}"
        obligation = invoke_llm(prompt)
        scratchpad = add_message(state, agent, f"Extracted obligation: {obligation}")
        return {"obligation_sentence": obligation, "scratchpad": scratchpad}

    def policy_theme_classifier(state: GraphState) -> GraphState:
        agent = "Theme Classifier"
        prompt = f"Classify the obligation into policy themes (e.g., AML, Risk, Cyber):\n{state['obligation_sentence']}"
        theme = invoke_llm(prompt)
        human_needed = True if theme.lower() in ["unclear", "multiple"] else False
        scratchpad = add_message(state, agent, f"Theme classified: {theme}")
        return {"policy_theme": theme, "human_intervention_needed": human_needed, "scratchpad": scratchpad}

    def human_intervention_node(state: GraphState) -> GraphState:
        agent = "Human-in-the-Loop"
        current = state.get("policy_theme")
        user_input = input(f"Review LLM classified theme '{current}'. Enter override or press ENTER: ").strip()
        updated = user_input if user_input else current
        scratchpad = add_message(state, agent, f"Final Theme: {updated}")
        return {"policy_theme": updated, "human_intervention_needed": False, "scratchpad": scratchpad}

    def responsible_party_locator(state: GraphState) -> GraphState:
        agent = "Responsible Party Locator"
        prompt = f"Who in a company is usually responsible for:\n{state['obligation_sentence']}"
        party = invoke_llm(prompt)
        scratchpad = add_message(state, agent, f"Responsible party: {party}")
        return {"responsible_party": party, "scratchpad": scratchpad}

    def compliance_timeline_extractor(state: GraphState) -> GraphState:
        agent = "Compliance Timeline Extractor"
        prompt = f"Extract timeline (Date, Type, Description) from:\n{state['raw_text']}"
        table = invoke_llm(prompt)
        scratchpad = add_message(state, agent, f"Compliance dates: {table}")
        return {"compliance_dates": table, "scratchpad": scratchpad}

    def divisional_impact_extractor(state: GraphState) -> GraphState:
        agent = "Divisional Impact Extractor"
        prompt = f"Extract table of division, region, impact, taxonomy, risk rating from:\n{state['raw_text']}"
        impact = invoke_llm(prompt)
        scratchpad = add_message(state, agent, f"Division impacts: {impact}")
        return {"divisional_impact": impact, "scratchpad": scratchpad}

    def ownership_info_extractor(state: GraphState) -> GraphState:
        agent = "Ownership Extractor"
        prompt = f"Extract ownership assessment from text. Format: Owner, Status, Assigned Date, Decision Date.\nText:\n{state['raw_text']}"
        ownership = invoke_llm(prompt)
        scratchpad = add_message(state, agent, f"Ownership: {ownership}")
        return {"ownership": ownership, "scratchpad": scratchpad}

    def summarizer(state: GraphState) -> GraphState:
        agent = "Summarizer"
        prompt = f"Summarize the requirement and key facts in 2 lines:\n\n{state['obligation_sentence']}\n{state['policy_theme']}\n{state['external_context']}"
        summary = invoke_llm(prompt)
        scratchpad = add_message(state, agent, f"Summary: {summary}")
        return {"summary": summary, "scratchpad": scratchpad}

    def check_for_human_intervention(state: GraphState):
        return "human_intervention_node" if state.get("human_intervention_needed") else "continue"

    workflow = StateGraph(GraphState, checkpointer=checkpointer)

    workflow.add_node("extractor", pdf_to_text_extractor)
    workflow.add_node("regulation_summary", regulatory_description_extractor)
    workflow.add_node("markdown", text_to_markdown_formatter)
    workflow.add_node("terms", regulatory_term_identifier)
    workflow.add_node("knowledge", external_knowledge_retriever)
    workflow.add_node("obligation", obligation_sentence_extractor)
    workflow.add_node("theme", policy_theme_classifier)
    workflow.add_node("human_intervention_node", human_intervention_node)
    workflow.add_node("responsible_party", responsible_party_locator)
    workflow.add_node("compliance_dates", compliance_timeline_extractor)
    workflow.add_node("divisional_impact", divisional_impact_extractor)
    workflow.add_node("ownership", ownership_info_extractor)
    workflow.add_node("summary", summarizer)

    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "regulation_summary")
    workflow.add_edge("regulation_summary", "markdown")
    workflow.add_edge("markdown", "terms")
    workflow.add_edge("terms", "knowledge")
    workflow.add_edge("knowledge", "obligation")
    workflow.add_edge("obligation", "theme")
    workflow.add_conditional_edges("theme", check_for_human_intervention, {
        "human_intervention_node": "human_intervention_node",
        "continue": "responsible_party"
    })
    workflow.add_edge("human_intervention_node", "responsible_party")
    workflow.add_edge("responsible_party", "compliance_dates")
    workflow.add_edge("compliance_dates", "divisional_impact")
    workflow.add_edge("divisional_impact", "ownership")
    workflow.add_edge("ownership", "summary")
    workflow.add_edge("summary", END)

    return workflow.compile()
