# dummy_llm.py

class DummyLLM:
    def invoke(self, prompt: str) -> str:
        if "extract_terms" in prompt:
            return "FCA, REP-CRIM, accounting date"
        elif "retrieve_context" in prompt:
            return "The FCA mandates REP-CRIM for financial crime monitoring."
        elif "extract_obligation" in prompt:
            return "Firms must submit REP-CRIM within 60 days of accounting date."
        elif "classify_theme" in prompt:
            return "Financial Crime"
        elif "find_party" in prompt:
            return "Compliance Department"
        elif "summarize" in prompt:
            return "Submit REP-CRIM to FCA to comply with financial crime policy."
        elif "division_function_identifier" in prompt:
            return "Compliance, Risk Management"
        elif "risk_scoring" in prompt:
            return "High"
        elif "qa_critic" in prompt:
            return "Reviewed. Obligation and classification look consistent."
        else:
            return "No output. Check prompt format."

# Example usage:
# from dummy_llm import DummyLLM
# llm = DummyLLM()
# print(llm.invoke("Thought: Identify key terms.\nAction: extract_terms\nAction Input: ..."))
