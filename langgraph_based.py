# First, make sure you have the necessary libraries:
# pip install -U langgraph langchain-openai

import os
import json
from typing import TypedDict, List, Callable, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI # Example LLM

# --- 1. Define the State for the Graph ---
# This is an expanded state based on your new requirements, now including reasoning fields.

class GraphState(TypedDict):
    """
    Represents the state of our graph, combining all required fields and their reasoning.

    Attributes:
        raw_text: The initial regulatory text input.
        regulatory_terms: A list of identified key terms.
        external_context: Knowledge retrieved from external sources.
        obligation_sentence: The extracted core obligation.
        
        policy_theme: The classified policy theme for the obligation.
        policy_theme_reasoning: The reasoning for the policy theme classification.
        
        responsible_party: The entity responsible for the obligation.
        responsible_party_reasoning: The reasoning for identifying the responsible party.
        
        divisional_impact: The identified impact on business divisions.
        divisional_impact_reasoning: The reasoning for the divisional impact assessment.

        risk_score: A calculated risk score (e.g., High, Medium, Low).
        risk_score_reasoning: The reasoning behind the assigned risk score.

        summary: A final summary of the analysis.
        qa_notes: Notes from a final QA/critic review.
        human_intervention_needed: A flag to signal a pause for user input.
        scratchpad: A list of thoughts and actions from the agent.
    """
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


# --- 2. The Main Workflow Builder ---
# This function encapsulates the entire graph creation process.

def build_workflow(llm: Callable, checkpointer=None):
    """
    Builds the LangGraph workflow with modular nodes and conditional edges.
    """

    # --- Helper for Scratchpad ---
    def add_to_scratchpad(state: GraphState, agent: str, thought: str, action_result: str) -> dict:
        """Adds a step to the scratchpad and returns the update dictionary."""
        current_scratchpad = state.get("scratchpad", [])
        new_entry = {
            "agent": agent,
            "thought": thought,
            "action_result": action_result
        }
        print(f"[{agent}] Thought: {thought}")
        # Pretty print JSON results for readability
        try:
            parsed_result = json.loads(action_result)
            print(f"[{agent}] Action Result:\n{json.dumps(parsed_result, indent=2)}")
        except json.JSONDecodeError:
            print(f"[{agent}] Action Result: {action_result}")
            
        return {"scratchpad": current_scratchpad + [new_entry]}

    # --- Tool Definitions (Nodes) ---
    # Each function is now a self-contained node in the graph.

    def extract_terms_node(state: GraphState) -> dict:
        prompt = f"From the following regulatory text, extract the key terms, names, and acronyms. Respond with a comma-separated list.\n\nText: {state['raw_text']}"
        thought = "I need to identify the key terms in the raw text."
        result = llm.invoke(prompt).content
        terms = [term.strip() for term in result.split(',')]
        
        state_update = add_to_scratchpad(state, "Term Identifier", thought, result)
        state_update["regulatory_terms"] = terms
        return state_update

    def retrieve_context_node(state: GraphState) -> dict:
        terms_str = ", ".join(state.get("regulatory_terms", []))
        prompt = f"Based on the key terms '{terms_str}', provide a brief, one-sentence external context. Imagine you are searching a knowledge base.\n\nTerms: {terms_str}"
        thought = "I will use the extracted terms to find relevant external context."
        result = llm.invoke(prompt).content
        
        state_update = add_to_scratchpad(state, "Context Retriever", thought, result)
        state_update["external_context"] = result
        return state_update

    def extract_obligation_node(state: GraphState) -> dict:
        prompt = f"From the following text, extract the core sentence or clause that describes the main obligation or requirement.\n\nText: {state['raw_text']}"
        thought = "I need to find the specific sentence that states what must be done."
        result = llm.invoke(prompt).content
        
        state_update = add_to_scratchpad(state, "Obligation Extractor", thought, result)
        state_update["obligation_sentence"] = result
        return state_update

    def classify_theme_node(state: GraphState) -> dict:
        prompt = f"Classify the following regulatory obligation into a concise policy theme (e.g., 'Financial Crime Prevention', 'Data Privacy'). Respond with a JSON object containing two keys: 'theme' and 'reasoning'.\n\nObligation: {state['obligation_sentence']}"
        thought = "I will classify the obligation into a high-level policy theme and explain my reasoning."
        result = llm.invoke(prompt).content
        parsed_result = json.loads(result)
        
        state_update = add_to_scratchpad(state, "Theme Classifier", thought, result)
        state_update["policy_theme"] = parsed_result.get("theme")
        state_update["policy_theme_reasoning"] = parsed_result.get("reasoning")
        state_update["human_intervention_needed"] = True # Signal for human check
        return state_update

    def find_party_node(state: GraphState) -> dict:
        prompt = f"Based on the following obligation, who is the responsible party or entity? Respond with a JSON object containing two keys: 'party' and 'reasoning'.\n\nObligation: {state['obligation_sentence']}"
        thought = "I need to identify which entity is responsible and explain why."
        result = llm.invoke(prompt).content
        parsed_result = json.loads(result)

        state_update = add_to_scratchpad(state, "Party Locator", thought, result)
        state_update["responsible_party"] = parsed_result.get("party")
        state_update["responsible_party_reasoning"] = parsed_result.get("reasoning")
        return state_update

    def identify_division_node(state: GraphState) -> dict:
        prompt = f"For the obligation '{state['obligation_sentence']}', which corporate divisions would be most impacted (e.g., 'Compliance, Finance')? Respond with a JSON object containing two keys: 'divisions' (a comma-separated string) and 'reasoning'."
        thought = "I need to determine which business divisions are affected and explain my logic."
        result = llm.invoke(prompt).content
        parsed_result = json.loads(result)
        
        state_update = add_to_scratchpad(state, "Division Identifier", thought, result)
        state_update["divisional_impact"] = parsed_result.get("divisions")
        state_update["divisional_impact_reasoning"] = parsed_result.get("reasoning")
        return state_update

    def score_risk_node(state: GraphState) -> dict:
        prompt = f"Given the obligation '{state['obligation_sentence']}' and theme '{state['policy_theme']}', assess the compliance risk. Respond with a JSON object containing two keys: 'score' (one word: High, Medium, or Low) and 'reasoning'."
        thought = "I will assess the risk level of this obligation and justify the score."
        result = llm.invoke(prompt).content
        parsed_result = json.loads(result)

        state_update = add_to_scratchpad(state, "Risk Scorer", thought, result)
        state_update["risk_score"] = parsed_result.get("score")
        state_update["risk_score_reasoning"] = parsed_result.get("reasoning")
        return state_update

    def summarize_node(state: GraphState) -> dict:
        context = json.dumps({k: v for k, v in state.items() if k not in ['scratchpad', 'raw_text', 'human_intervention_needed']})
        prompt = f"Create a final, one-paragraph summary based on the following analysis data:\n\n{context}"
        thought = "I will now generate the final summary of the analysis."
        result = llm.invoke(prompt).content
        
        state_update = add_to_scratchpad(state, "Summarizer", thought, result)
        state_update["summary"] = result
        return state_update

    def qa_critic_node(state: GraphState) -> dict:
        review_state = {k: v for k, v in state.items() if k not in ['scratchpad', 'human_intervention_needed']}
        prompt = f"You are a QA critic. Review the following JSON analysis of a regulatory text. Check for inconsistencies or logical errors between the conclusions and their reasoning. Provide brief, actionable notes. If it looks good, say 'Analysis appears consistent and complete.'\n\nAnalysis:\n{json.dumps(review_state, indent=2)}"
        thought = "I will now perform a final QA check on the entire analysis to ensure consistency and completeness."
        result = llm.invoke(prompt).content
        
        state_update = add_to_scratchpad(state, "QA Critic", thought, result)
        state_update["qa_notes"] = result
        return state_update

    # --- Human-in-the-Loop Logic ---
    def check_for_human_intervention(state: GraphState):
        return "human_intervention_node" if state.get("human_intervention_needed") else "continue"

    def human_intervention_node(state: GraphState):
        current_theme = state.get("policy_theme")
        print("\n" + "="*50)
        print("HUMAN-IN-THE-LOOP CHECKPOINT")
        print(f"The agent classified the policy theme as: '{current_theme}'")
        user_input = input("Press ENTER to approve, or type a new theme and press ENTER: ")

        if user_input.strip():
            print(f"User has overridden the theme to: '{user_input.strip()}'")
            return {"policy_theme": user_input.strip(), "human_intervention_needed": False}
        else:
            print("User approved the theme.")
            return {"human_intervention_needed": False}

    # --- Build the Graph ---
    workflow = StateGraph(GraphState)

    workflow.add_node("extract_terms", extract_terms_node)
    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("extract_obligation", extract_obligation_node)
    workflow.add_node("classify_theme", classify_theme_node)
    workflow.add_node("human_intervention", human_intervention_node)
    workflow.add_node("find_party", find_party_node)
    workflow.add_node("identify_division", identify_division_node)
    workflow.add_node("score_risk", score_risk_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("qa_critic", qa_critic_node)

    workflow.set_entry_point("extract_terms")
    workflow.add_edge("extract_terms", "retrieve_context")
    workflow.add_edge("retrieve_context", "extract_obligation")
    workflow.add_edge("extract_obligation", "classify_theme")

    workflow.add_conditional_edges(
        "classify_theme",
        check_for_human_intervention,
        {"continue": "find_party", "human_intervention_node": "human_intervention"}
    )
    
    workflow.add_edge("human_intervention", "find_party")
    workflow.add_edge("find_party", "identify_division")
    workflow.add_edge("identify_division", "score_risk")
    workflow.add_edge("score_risk", "summarize")
    workflow.add_edge("summarize", "qa_critic")
    workflow.add_edge("qa_critic", END)

    return workflow.compile(checkpointer=checkpointer)


# --- Example Usage ---
if __name__ == "__main__":
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    except ImportError:
        print("Please install langchain-openai: pip install langchain-openai")
        exit()
    except Exception as e:
        print(f"Could not initialize ChatOpenAI. Make sure your OPENAI_API_KEY is set. Error: {e}")
        exit()

    app = build_workflow(llm)

    initial_input = {
        "raw_text": "The Financial Conduct Authority (FCA) requires all investment firms to submit their annual financial crime report (REP-CRIM) within 60 business days of the firm's accounting reference date.",
        "scratchpad": []
    }

    print("Starting Regulatory Analysis Workflow...")
    print("="*50 + "\n")

    final_state = None
    for s in app.stream(initial_input, stream_mode="values"):
        final_state = s

    print("\n" + "="*50)
    print("Workflow Complete. Final Analysis State:")
    print("="*50)
    final_analysis = {k: v for k, v in final_state.items() if k != 'scratchpad'}
    print(json.dumps(final_analysis, indent=2))
