# regulatory_analysis_graph.py
# pip install -U langgraph

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

# --- Tools (Nodes) ---
def build_workflow(llm: Callable, checkpointer=None):

    def invoke_llm(prompt: str) -> str:
        return llm.invoke(prompt)

    def pdf_to_text_extractor(state: GraphState) -> GraphState:
        print("--- PDF-to-Text Extractor ---")
        return {"raw_text": state["raw_text"]}

    def text_to_markdown_formatter(state: GraphState) -> GraphState:
        print("--- Text-to-Markdown Formatter ---")
        markdown = f"## Regulatory Requirement\n\n{state['raw_text']}"
        return {"markdown_text": markdown}

    def regulatory_term_identifier(state: GraphState) -> GraphState:
        print("--- Regulatory Term Identifier ---")
        prompt = f"Identify all key regulatory entities, terms, codes, and dates from the following text:\n\n{state['raw_text']}"
        terms = invoke_llm(prompt)
        return {"regulatory_terms": terms.split(", ")}

    def external_knowledge_retriever(state: GraphState) -> GraphState:
        print("--- External Knowledge Retriever ---")
        query = ", ".join(state["regulatory_terms"])
        prompt = f"Using internal whitepapers and knowledge base, retrieve relevant context for the following terms:\n{query}"
        context = invoke_llm(prompt)
        return {"external_context": context}

    def obligation_sentence_extractor(state: GraphState) -> GraphState:
        print("--- Obligation Sentence Extractor ---")
        prompt = f"From the following text, extract the core obligation clause (look for 'must', 'shall', 'required'):\n{state['raw_text']}"
        obligation = invoke_llm(prompt)
        return {"obligation_sentence": obligation}

    def policy_theme_classifier(state: GraphState) -> GraphState:
        print("--- Policy Theme Classifier ---")
        prompt = f"Classify the following obligation into a policy theme (e.g., AML, Cybersecurity, Operational Risk):\n{state['obligation_sentence']}"
        theme = invoke_llm(prompt)
        return {"policy_theme": theme, "human_intervention_needed": True}

    def responsible_party_locator(state: GraphState) -> GraphState:
        print("--- Responsible Party Locator ---")
        prompt = f"Who in an organization is typically responsible for the following obligation:\n{state['obligation_sentence']}"
        party = invoke_llm(prompt)
        return {"responsible_party": party}

    def obligation_owner_mapper(state: GraphState) -> GraphState:
        print("--- Obligation-Owner Mapper ---")
        return {}

    def summarizer(state: GraphState) -> GraphState:
        print("--- Summarizer ---")
        prompt = f"Summarize this compliance requirement and its implications in 2 lines:\n\nRequirement: {state['obligation_sentence']}\nTheme: {state['policy_theme']}\nContext: {state['external_context']}"
        summary = invoke_llm(prompt)
        return {"summary": summary}

    def check_for_human_intervention(state: GraphState):
        return "human_intervention_node" if state.get("human_intervention_needed") else "continue"

    def human_intervention_node(state: GraphState):
        print("--- HUMAN CHECK ---")
        current = state.get("policy_theme")
        user_input = input(f"LLM classified theme as '{current}'. Press ENTER to approve or type a new one: ").strip()
        if user_input:
            return {"policy_theme": user_input, "human_intervention_needed": False}
        return {"human_intervention_needed": False}

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
