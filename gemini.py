import asyncio
import json
import logging
import re
from typing import TypedDict, List, Dict, Any, Literal

import websockets
from websockets.exceptions import ConnectionClosed

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Mock LLM and Parser for Standalone Execution ---
# In a real application, these would be your actual LangChain LLM and parser calls.
class MockLLM:
    async def invoke(self, prompt: str, documentLabels: List[str] = None) -> str:
        logging.info(f"MockLLM invoked with prompt starting with: {prompt[:100]}...")
        # Simulate LLM thinking time
        await asyncio.sleep(1)
        # Return a plausible JSON string based on the prompt's "Action"
        if "Extract Terms" in prompt:
            return '{"Action Result": {"terms": ["Regulatory Term 1", "Compliance Standard 2"]}}'
        if "Retrieve Context" in prompt:
            return '{"Action Result": {"context": "This regulation applies to all financial institutions."}}'
        if "Extract Obligation" in prompt:
            return '{"Action Result": {"identified_document": "Doc-123", "obligation_sentence": "All institutions must report their quarterly earnings.", "nature_of_obligation": "direct"}}'
        if "IRR Topic Classifier" in prompt:
            return '{"Action Result": {"irr_topic": "Financial Reporting", "reasoning": "The obligation is about reporting earnings."}}'
        if "Regional Topic Classifier" in prompt:
            return '{"Action Result": {"region_topic": "North America", "reasoning": "Applies to US-based institutions."}}'
        if "Party Locator" in prompt:
             return '{"Action Result": {"party": "Compliance Department", "reasoning": "Compliance is responsible for reports."}}'
        if "Division Identifier" in prompt:
            return '{"Action Result": {"divisions": ["Finance", "Accounting"], "reasoning": "These divisions handle financial data."}}'
        if "Risk Scorer" in prompt:
            return '{"Action Result": {"score_risk": "Medium", "score": "7/10", "reasoning": "Failure to comply has moderate penalties."}}'
        if "Timeline Extractor" in prompt:
            return '{"Action Result": {"timelines": [{"milestone": "Q3 Reporting Deadline", "deadline": "2025-09-30"}], "reasoning": "The deadline is specified in section 5.2."}}'
        if "Summarizer" in prompt:
            return '{"Action Result": {"summary": "This is a comprehensive summary of the regulatory analysis."}}'
        if "QA Critic" in prompt:
            return '{"Action Result": {"qa_notes": "The analysis appears consistent and logical."}}'
        if "User Query" in prompt:
            return '{"Action Result": {"answer": "Based on the analysis, the main deadline is September 30, 2025."}}'
        return '{"Action Result": {}}'

llm = MockLLM()

async def parse_react_response(agent: str, response: str) -> dict:
    logging.info(f"Parsing response for agent: {agent}")
    try:
        # In a real scenario, you might have more complex parsing logic.
        # Here, we just parse the JSON string from the mock LLM.
        parsed_json = json.loads(response)
        return parsed_json.get("Action Result", {})
    except json.JSONDecodeError as e:
        logging.error(f"JSONDecodeError in parse_react_response: {e}")
        return {}

# --- State Definition ---
class GraphState(TypedDict):
    """Represents the state of our graph."""
    documentLabels: List[str]
    title: str
    summary: str
    regulatory_terms: List[str]
    external_context: str
    obligation_sentence: str
    identified_document: str
    nature_of_obligation: str
    irr_topic: str
    irr_topic_reasoning: str
    region_topic: str
    region_topic_reasoning: str
    responsible_party: str
    responsible_party_reasoning: str
    divisional_impact: List[str]
    divisional_impact_reasoning: str
    risk_score: str
    risk_score_reasoning: str
    timelines: List[Dict]
    timelines_reasoning: str
    qa_notes: str
    human_intervention_needed: bool
    json_output: dict
    scratchpad: List[str]


# --- Utility Functions ---

def convert_json_to_md(json_data: Any, indent: int = 0) -> str:
    """Recursively converts a JSON object to a Markdown string for better readability."""
    if not json_data:
        return "No data available."

    md_content = ""
    indent_str = "  " * indent  # Indentation for nested structures

    if isinstance(json_data, dict):
        for key, value in json_data.items():
            md_content += f"{indent_str}- **{key}:**\n"
            md_content += convert_json_to_md(value, indent + 1)
    elif isinstance(json_data, list):
        for item in json_data:
            md_content += convert_json_to_md(item, indent + 1)
    else:
        md_content += f"{indent_str}- {json_data}\n"

    return md_content

async def update_state(state: GraphState, agent: str, response: str, update_fields: dict, websocket=None) -> GraphState:
    """Updates the state after an agent's execution and sends the update."""
    new_scratchpad = state.get('scratchpad', [])
    new_scratchpad.append(response)
    
    parsed_content = await parse_react_response(agent, response)
    
    updated_state = state.copy()
    updated_state['scratchpad'] = new_scratchpad
    updated_state['json_output'] = parsed_content

    for state_key, parsed_key in update_fields.items():
        if parsed_key in parsed_content:
            updated_state[state_key] = parsed_content[parsed_key]

    if websocket:
        await websocket.send(json.dumps({
            "type": "progress",
            "payload": {
                "agent": agent,
                "thinking": "Parsed response and updated state.",
                "content": parsed_content
            }
        }))

    return updated_state

# --- Graph Nodes ---

async def extract_terms_node(state: GraphState, websocket=None) -> GraphState:
    """Extracts key regulatory terms from the summary."""
    prompt = f"""You are a compliance analyst AI. Based on the regulatory summary, identify key regulatory terms.
    Regulatory Summary Text: {state.get('summary', '')}
    Respond in JSON Format: {{"Action": "Extract Terms", "Action Result": {{"terms": ["term1", "term2"]}}}}
    """
    response = await llm.invoke(prompt, documentLabels=state.get("documentLabels", []))
    return await update_state(state, "Term Identifier", response, {"regulatory_terms": "terms"}, websocket)

async def retrieve_context_node(state: GraphState, websocket=None) -> GraphState:
    """Retrieves external context for the identified terms."""
    prompt = f"""You are a compliance analyst AI. Based on the key terms {state.get('regulatory_terms', [])}, provide relevant external context.
    Respond in JSON Format: {{"Action": "Retrieve Context", "Action Result": {{"context": "..."}}}}
    """
    response = await llm.invoke(prompt, documentLabels=state.get("documentLabels", []))
    return await update_state(state, "Context Retriever", response, {"external_context": "context"}, websocket)

async def extract_obligation_node(state: GraphState, websocket=None) -> GraphState:
    """Extracts the main obligation sentence from the regulatory text."""
    prompt = f"""You are a compliance analyst AI. Based on the regulatory summary, identify the most relevant regulatory document and extract the main obligation sentence.
    Focus on terms such as 'shall', 'should', 'must', 'required', etc., to determine obligations.
    Clarify whether the obligation is a direct mandate imposed by the regulation or a derived implication of compliance.
    Regulatory Summary: {state.get('summary', '')}
    Respond in JSON Format: {{"Action": "Extract Obligation", "Action Result": {{"identified_document": "...", "obligation_sentence": "...", "nature_of_obligation": "..."}}}}
    """
    response = await llm.invoke(prompt, documentLabels=state.get("documentLabels", []))
    return await update_state(state, "Obligation Extractor", response, {
        "identified_document": "identified_document",
        "obligation_sentence": "obligation_sentence",
        "nature_of_obligation": "nature_of_obligation"
    }, websocket)

async def classify_irr_topic_node(state: GraphState, websocket=None) -> GraphState:
    """Classifies the obligation into an IRR Topic."""
    # In a real app, irr_topics would come from a file or database
    irr_topics = "['Financial Crime', 'Reporting', 'Data Privacy', 'Market Conduct']"
    prompt = f"""You are a compliance analyst AI. Classify the following obligation into one of the IRR Topics listed below.
    IRR Topics: {irr_topics}
    Obligation: {state.get('obligation_sentence', '')}
    Respond in JSON Format: {{"Action": "IRR Topic Classifier", "Action Result": {{"irr_topic": "...", "reasoning": "..."}}}}
    """
    response = await llm.invoke(prompt, documentLabels=state.get("documentLabels", []))
    return await update_state(state, "IRR Topic Classifier", response, {
        "irr_topic": "irr_topic",
        "irr_topic_reasoning": "reasoning"
    }, websocket)

async def summarize_node(state: GraphState, websocket=None) -> GraphState:
    """Generates a final, comprehensive markdown summary."""
    # This node constructs a detailed summary from all the data gathered in the state.
    summary_prompt = f"""
    You are a compliance analyst AI. Generate a comprehensive Markdown-formatted summary based on the analysis performed.
    
    ### Comprehensive Regulatory Analysis Summary
    **Regulation Title:** {state.get('title', 'N/A')}
    **Document Labels:** {', '.join(state.get('documentLabels', []))}
    
    #### Regulatory Terms
    {', '.join(state.get('regulatory_terms', ['No terms identified.']))}
    
    #### External Context
    {state.get('external_context', 'No external context provided.')}
    
    #### Obligation
    **Sentence:** {state.get('obligation_sentence', 'No obligation sentence identified.')}
    **Nature:** {state.get('nature_of_obligation', 'N/A')}
    
    #### Timelines
    {convert_json_to_md(state.get('timelines', []))}
    **Reasoning:** {state.get('timelines_reasoning', 'No reasoning provided.')}
    
    #### Policy Theme
    **Theme:** {state.get('irr_topic', 'No theme identified.')}
    **Reasoning:** {state.get('irr_topic_reasoning', 'No reasoning provided.')}
    
    #### Responsible Party
    **Party:** {state.get('responsible_party', 'No party identified.')}
    **Reasoning:** {state.get('responsible_party_reasoning', 'No reasoning provided.')}
    
    #### Divisional Impact
    **Divisions:** {', '.join(state.get('divisional_impact', ['No divisions identified.']))}
    **Reasoning:** {state.get('divisional_impact_reasoning', 'No reasoning provided.')}
    
    #### Risk Assessment
    **Score:** {state.get('risk_score', 'No risk score provided.')}
    **Reasoning:** {state.get('risk_score_reasoning', 'No reasoning provided.')}
    
    #### QA Notes
    {state.get('qa_notes', 'No QA notes.')}

    Respond in JSON Format: {{"Action": "Summarizer", "Action Result": {{"summary": "Markdown formatted summary here..."}}}}
    """
    response = await llm.invoke(summary_prompt)
    return await update_state(state, "Summarizer", response, {"summary": "summary"}, websocket)

async def check_for_human_intervention(state: GraphState) -> Literal["continue", "intervene"]:
    """Checks if human intervention is needed."""
    if state.get("human_intervention_needed", False):
        logging.info("Human intervention is required. Routing to intervention node.")
        return "intervene"
    logging.info("No human intervention required. Continuing workflow.")
    return "continue"

async def human_intervention_node(state: GraphState, websocket=None) -> GraphState:
    """Pauses the workflow and waits for human feedback."""
    logging.info("Entering human intervention node. Awaiting feedback from client.")
    
    if not websocket or websocket.closed:
        logging.warning("WebSocket is not available for human intervention. Returning original state.")
        return state

    # Send the current state to the UI for review and modification
    await websocket.send(json.dumps({
        "type": "human_intervention_required",
        "payload": {
            "message": "Human intervention required. Please review and modify the data as needed.",
            "currentState": state
        }
    }))

    try:
        # Wait for the feedback message from the client
        feedback_str = await asyncio.wait_for(websocket.recv(), timeout=300.0) # 5-minute timeout
        feedback = json.loads(feedback_str)

        if feedback.get("type") == "human_feedback":
            logging.info("Human feedback received. Updating state.")
            updated_data = feedback.get("payload", {})
            
            # Create a new state object with the updates
            new_state = state.copy()
            new_state.update(updated_data)
            
            # Mark intervention as resolved and continue
            new_state["human_intervention_needed"] = False
            
            await websocket.send(json.dumps({
                "type": "info",
                "payload": "Feedback received. Resuming analysis..."
            }))
            
            return new_state
        else:
            logging.warning("Received unexpected message during intervention. Ignoring.")
            await websocket.send(json.dumps({
                "type": "error",
                "payload": "Unexpected message received during intervention. Please send feedback or restart."
            }))
            return state

    except asyncio.TimeoutError:
        logging.error("Timeout waiting for human intervention feedback.")
        await websocket.send(json.dumps({"type": "error", "payload": "Timeout: No feedback received."}))
        return state
    except json.JSONDecodeError:
        logging.error("Error decoding feedback from client.")
        await websocket.send(json.dumps({"type": "error", "payload": "Invalid JSON format for feedback."}))
        return state
    except Exception as e:
        logging.error(f"Error during human intervention: {e}")
        await websocket.send(json.dumps({"type": "error", "payload": f"An error occurred: {e}"}))
        return state

async def handle_user_query(state: GraphState, query: str, websocket=None):
    """Handles a one-off user query against the current state."""
    logging.info(f"Handling user query: {query}")
    prompt = f"""You are a helpful AI assistant. Based on the following analysis data, answer the user's query.
    
    Analysis Data:
    {json.dumps(state, indent=2)}

    User Query: {query}

    Respond in JSON Format: {{"Action": "User Query", "Action Result": {{"answer": "Your answer here..."}}}}
    """
    response_str = await llm.invoke(prompt)
    parsed_response = await parse_react_response("User Query Handler", response_str)
    answer = parsed_response.get("answer", "Sorry, I could not find an answer to your question.")

    if websocket:
        await websocket.send(json.dumps({
            "type": "query_answer",
            "payload": {
                "query": query,
                "answer": answer
            }
        }))

# --- Main WebSocket Handler ---

async def handler(websocket, path):
    """Main WebSocket connection handler."""
    logging.info(f"Client connected from {websocket.remote_address}")
    current_state = {}

    try:
        # This loop handles all incoming messages from a single client
        async for message_str in websocket:
            try:
                message = json.loads(message_str)
                msg_type = message.get("type")
                payload = message.get("payload")

                if msg_type == "start_analysis":
                    logging.info(f"Received start_analysis request: {payload}")
                    
                    # 1. Initialize State
                    if "::" not in payload:
                        await websocket.send(json.dumps({"type": "error", "payload": "Invalid format. Use 'REG ID :: SUBJECT'"}))
                        continue
                    
                    reg_id, title = payload.split("::", 1)
                    current_state = {
                        "documentLabels": [reg_id.strip()],
                        "title": title.strip(),
                        "summary": title.strip(), # Use title as initial summary
                        "regulatory_terms": [],
                        "external_context": "",
                        "obligation_sentence": "",
                        "identified_document": "",
                        "nature_of_obligation": "",
                        "irr_topic": "",
                        "irr_topic_reasoning": "",
                        "region_topic": "",
                        "region_topic_reasoning": "",
                        "responsible_party": "",
                        "responsible_party_reasoning": "",
                        "divisional_impact": [],
                        "divisional_impact_reasoning": "",
                        "risk_score": "",
                        "risk_score_reasoning": "",
                        "timelines": [],
                        "timelines_reasoning": "",
                        "qa_notes": "",
                        "human_intervention_needed": True, # Start with intervention needed
                        "json_output": {},
                        "scratchpad": [],
                    }
                    await websocket.send(json.dumps({"type": "info", "payload": f"State initialized for {reg_id.strip()}"}))

                    # 2. Run the full analysis workflow
                    # In a real LangGraph app, you would define the graph and stream it.
                    # Here we simulate the flow by calling nodes sequentially.
                    
                    # Simulating a graph flow
                    current_state = await extract_terms_node(current_state, websocket)
                    current_state = await retrieve_context_node(current_state, websocket)
                    current_state = await extract_obligation_node(current_state, websocket)
                    
                    # Conditional step for human intervention
                    intervention_choice = await check_for_human_intervention(current_state)
                    if intervention_choice == "intervene":
                        # The human_intervention_node will now handle the pause and wait for a "human_feedback" message
                        current_state = await human_intervention_node(current_state, websocket)
                    
                    # Continue the flow after potential intervention
                    current_state = await classify_irr_topic_node(current_state, websocket)
                    current_state = await summarize_node(current_state, websocket)
                    
                    await websocket.send(json.dumps({
                        "type": "analysis_complete",
                        "payload": current_state
                    }))

                elif msg_type == "user_query":
                    if not current_state:
                        await websocket.send(json.dumps({"type": "error", "payload": "Analysis not started. Cannot answer query."}))
                        continue
                    await handle_user_query(current_state, payload, websocket)
                
                # The "human_feedback" message is handled inside the human_intervention_node,
                # which is called during the "start_analysis" flow.

                else:
                    logging.warning(f"Unknown message type received: {msg_type}")
                    await websocket.send(json.dumps({"type": "error", "payload": f"Unknown message type: {msg_type}"}))

            except json.JSONDecodeError:
                logging.error(f"Could not decode JSON from message: {message_str}")
                await websocket.send(json.dumps({"type": "error", "payload": "Invalid JSON format."}))
            except Exception as e:
                logging.error(f"An error occurred in the handler loop: {e}", exc_info=True)
                await websocket.send(json.dumps({"type": "error", "payload": f"An unexpected server error occurred: {e}"}))

    except ConnectionClosed:
        logging.info(f"Client {websocket.remote_address} disconnected.")
    except Exception as e:
        logging.error(f"The WebSocket handler terminated with an error: {e}", exc_info=True)
    finally:
        logging.info(f"Connection with {websocket.remote_address} closed.")


# --- Server Startup ---

async def main():
    """Starts the WebSocket server."""
    port = 8765
    logging.info(f"Starting WebSocket server on ws://localhost:{port}")
    async with websockets.serve(handler, "localhost", port):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Server is shutting down.")

