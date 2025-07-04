# run_graph.py

from central_mind_router import build_workflow
from langgraph.checkpoint import FileCheckpointer
import os

# --- Simulated LLM wrapper ---
class MockLLM:
    def invoke(self, prompt: str) -> str:
        print(f"\n[LLM Prompt]\n{prompt}\n")
        return "Mocked response based on prompt"

# --- Setup ---
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpointer = FileCheckpointer(checkpoint_dir)

llm = MockLLM()
app = build_workflow(llm, checkpointer=checkpointer)

# --- Sample Input ---
initial_input = {
    "raw_text": (
        "The Financial Conduct Authority (FCA) requires all investment firms to submit their "
        "annual financial crime report (REP-CRIM) within 60 business days of the firm's accounting reference date."
    )
}

# --- Run ---
print("\n==== Starting Regulatory Analysis ====")
final_state = None
for step in app.stream(initial_input):
    node = list(step.keys())[0]
    print(f"\n--- Node: {node} ---")
    print(step[node])
    final_state = step[node]

print("\n==== FINAL OUTPUT ====")
print(f"Obligation: {final_state.get('obligation_sentence')}")
print(f"Theme: {final_state.get('policy_theme')}")
print(f"Owner: {final_state.get('responsible_party')}")
print(f"Context: {final_state.get('external_context')}")
print(f"Summary: {final_state.get('summary')}")
