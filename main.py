from langgraph.checkpoint import FileCheckpointer
import os

# Set up a file-based checkpointer
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpointer = FileCheckpointer(checkpoint_dir)

# Inject it into the workflow
app = build_workflow(llm, checkpointer=checkpointer)
