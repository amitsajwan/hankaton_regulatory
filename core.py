# regulatory_analysis_graph.py
# pip install -U langgraph

import os
import json
from typing import TypedDict, List, Callable
from langgraph.graph import StateGraph, END

# --- Define the State ---
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
    scratchpad: dict

# --- Tools (Nodes) ---
def build_workflow(llm: Callable, checkpointer=None):

    def invoke_llm(prompt: str) -> str:
        return llm.invoke(prompt)

    def add_message(state: GraphState, msg: str):
        scratchpad = state.get("scratchpad", {})
        scratchpad.setdefault("intermediate_messages", []).append(msg)
        return scratchpad

    def pdf_to_text_extractor(state: GraphState) -> GraphState:
        msg = "--- PDF-to-Text Extractor ---"
        print(msg)
        scratchpad = add_message(state, msg)
        return {"raw_text": state["raw_text"], "scratchpad": scratchpad}

    def text_to_markdown_formatter(state: GraphState) -> GraphState:
        msg = "--- Text-to-Markdown Formatter ---"
        print(msg)
        markdown = f"## Regulatory Requirement\n\n{state['raw_text']}"
        scratchpad = add_message(state, msg)
        return {"markdown_text": markdown, "scratchpad": scratchpad}

    def regulatory_term_identifier(state: GraphState) -> GraphState:
        msg = "--- Regulatory Term Identifier ---"
        print(msg)
        prompt = f"Identify all key regulatory entities, terms, codes, and dates from the following text:\n\n{state['raw_text']}"
        terms = invoke_llm(prompt)
        scratchpad = add_message(state, f"Identified terms: {terms}")
        return {"regulatory_terms": terms.split(", "), "scratchpad": scratchpad}

    def external_knowledge_retriever(state: GraphState) -> GraphState:
        msg = "--- External Knowledge Retriever ---"
        print(msg)
        query = ", ".join(state["regulatory_terms"])
        prompt = f"Using internal whitepapers and knowledge base, retrieve relevant context for the following terms:\n{query}"
        context = invoke_llm(prompt)
        scratchpad = add_message(state, f"Retrieved context: {context}")
        return {"external_context": context, "scratchpad": scratchpad}

    def obligation_sentence_extractor(state: GraphState) -> GraphState:
        msg = "--- Obligation Sentence Extractor ---"
        print(msg)
        prompt = f"From the following text, extract the core obligation clause (look for 'must', 'shall', 'required'):\n{state['raw_text']}"
        obligation = invoke_llm(prompt)
        scratchpad = add_message(state, f"Extracted obligation: {obligation}")
        return {"obligation_sentence": obligation, "scratchpad": scratchpad}

    def policy_theme_classifier(state: GraphState) -> GraphState:
        msg = "--- Policy Theme Classifier ---"
        print(msg)
        prompt = f"Classify the following obligation into a policy theme (e.g., AML, Cybersecurity, Operational Risk):\n{state['obligation_sentence']}"
        theme = invoke_llm(prompt)
        scratchpad = add_message(state, f"Classified policy theme: {theme}")
        return {"policy_theme": theme, "human_intervention_needed": True, "scratchpad": scratchpad}

    def responsible_party_locator(state: GraphState) -> GraphState:
        msg = "--- Responsible Party Locator ---"
        print(msg)
        prompt = f"Who in an organization is typically responsible for the following obligation:\n{state['obligation_sentence']}"
        party = invoke_llm(prompt)
        scratchpad = add_message(state, f"Located responsible party: {party}")
        return {"responsible_party": party, "scratchpad": scratchpad}

    def obligation_owner_mapper(state: GraphState) -> GraphState:
        msg = "--- Obligation-Owner Mapper ---"
        print(msg)
        scratchpad = add_message(state, msg)
        return {"scratchpad": scratchpad}

    def summarizer(state: GraphState) -> GraphState:
        msg = "--- Summarizer ---"
        print(msg)
        prompt = f"Summarize this compliance requirement and its implications in 2 lines:\n\nRequirement: {state['obligation_sentence']}\nTheme: {state['policy_theme']}\nContext: {state['external_context']}"
        summary = invoke_llm(prompt)
        scratchpad = add_message(state, f"Generated summary: {summary}")
        return {"summary": summary, "scratchpad": scratchpad}

    def check_for_human_intervention(state: GraphState):
        return "human_intervention_node" if state.get("human_intervention_needed") else "continue"

    def human_intervention_node(state: GraphState):
        print("--- HUMAN CHECK ---")
        current = state.get("policy_theme")
        user_input = input(f"LLM classified theme as '{current}'. Press ENTER to approve or type a new one: ").strip()
        new_theme = user_input if user_input else current
        scratchpad = add_message(state, f"Final theme (after human check): {new_theme}")
        return {"policy_theme": new_theme, "human_intervention_needed": False, "scratchpad": scratchpad}

    workflow = StateGraph(GraphState, checkpointer=checkpointer)
    workflow.add_node("extractor", pdf_to_text_extractor)
    workflow.add_node("formatter", text_to_markdown_formatter)
    workflow.add_node("term_identifier", regulatory_term_identifier)
    workflow.add_node("retriever", external_knowledge_retriever)
    workflow.add_node("obligation_extractor", obligation_sentence_extractor)
    workflow.add_node("theme_classifier", policy_theme_classifier)
    workflow.add_node("party_locator", responsible_party_locator)
    workflow.add_node("mapper", obligation_owner_mapper)
    workflow.add_node("summarizer", summarizer)
    workflow.add_node("human_intervention_node", human_intervention_node)

    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "formatter")
    workflow.add_edge("formatter", "term_identifier")
    workflow.add_edge("term_identifier", "retriever")
    workflow.add_edge("retriever", "obligation_extractor")
    workflow.add_edge("obligation_extractor", "theme_classifier")
    workflow.add_conditional_edges(
        "theme_classifier",
        check_for_human_intervention,
        {
            "human_intervention_node": "human_intervention_node",
            "continue": "party_locator",
        }
    )
    workflow.add_edge("human_intervention_node", "party_locator")
    workflow.add_edge("party_locator", "mapper")
    workflow.add_edge("mapper", "summarizer")
    workflow.add_edge("summarizer", END)

    return workflow.compile()
