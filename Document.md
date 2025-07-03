# AI-Powered Regulatory Impact System – Full Hackathon Reference

## 🎯 Project Objective

Develop a GenAI-powered system that starts from regulatory feeds and outputs structured, actionable impact analysis across divisions, services, and geographies. Designed for fast execution during a hackathon, it leverages prompting and modular tools.

---

## 🧱 System Architecture (Micro-Tool Breakdown)

### Phase 1: Ingestion & Structuring

| Tool                 | Purpose                                             | Input           |
| -------------------- | --------------------------------------------------- | --------------- |
| `PDFTextExtractor`   | Extract raw text from RGS+ PDFs or HTML feeds       | RGS+, PDF, HTML |
| `WhitepaperEmbedder` | Embed whitepapers, SOPs, control docs for retrieval | Internal docs   |
| `MarkdownFormatter`  | Structure regulation into clean markdown sections   | Raw text        |

---

### Phase 2: Enrichment (Prompt-Based)

| Tool                    | Purpose                               | Output                  | Prompt?       |
| ----------------------- | ------------------------------------- | ----------------------- | ------------- |
| `ObligationExtractor`   | Extract "shall/must" sentences        | Clauses                 | ✅             |
| `ThemeClassifier`       | Assign LRR themes/taxonomy            | e.g. "Model Risk"       | ✅             |
| `RegionTagger`          | Detect jurisdictions (EU, APAC, etc.) | e.g. EU, US             | ✅             |
| `DivisionMapper`        | Map obligation to division/service    | e.g. Risk Ops           | ✅ + Embedding |
| `UrgencyScorer`         | Classify urgency from clause          | T+1, Future             | ✅             |
| `ImpactLevelClassifier` | Rate criticality of impact            | High, Medium, Low       | ✅             |
| `OwnerRoleSuggester`    | Suggest accountable function          | e.g. Head of Compliance | ✅             |

---

### Phase 3: Visualization & Outputs

| Tool                 | Purpose                                                |
| -------------------- | ------------------------------------------------------ |
| `HeatmapBuilder`     | Create impact matrix view across clauses vs. divisions |
| `RoadmapTimeline`    | Show upcoming regulation deadlines on a timeline       |
| `SummaryGenerator`   | Executive-level text summarization of regulation       |
| `StructuredExporter` | Export final output as JSON / CSV                      |

---

## 💡 Why Prompting is Best for Hackathon

* ✅ No model training required
* ✅ Fast iteration
* ✅ Zero deployment complexity
* ✅ Easy to tune outputs using prompt design

---

## 📌 Example Output JSON

```json
{
  "clause": "Firms must submit liquidity risk reports daily to the regulator.",
  "theme": "Market Risk",
  "region": "EU",
  "division": "Liquidity Risk",
  "impact": "High",
  "urgency": "T+1",
  "owner": "Risk Control Head"
}
```

---

## 🧠 Prompt Example – Theme Classification

**Instruction:**

> Classify this regulation clause into one of the following themes: Market Risk, AI Risk, Cybersecurity, AML, Corporate Governance.

**Input:**

> "All firms using algorithmic models must document audit trails for 3 years."

**Output:**

> "AI Risk"

---

## 📊 Interactive Heatmap Input Fields

* Clause
* Theme
* Division / Service
* Impact level
* Region
* Status (Draft/Final)
* Urgency (T+1 etc.)

Used to build dashboards with filters across business units and jurisdictions.

---

## 🗂️ Folder Layout

```
reg-impact-ai/
├── data/
│   ├── rgs_clauses.csv
│   ├── service_catalog.csv
├── prompts/
│   └── classify_theme.txt
├── tools/
│   ├── extract_clauses.py
│   ├── classify_prompt.py
│   ├── enrich_with_embeddings.py
├── app/
│   └── heatmap_dashboard.py
```

---

## ✅ Summary

| Area          | Tool/Approach                              |
| ------------- | ------------------------------------------ |
| Enrichment    | GPT-4o prompt-based classification         |
| Mapping       | Embedding + similarity to internal catalog |
| Visualization | Streamlit or Dash UI with filters          |
| Export        | Structured JSON / CSV for downstream       |

---

## 🚀 Hackathon Execution Plan

1. Parse regulation feeds (RGS+/PDF)
2. Run prompt-based tools for enrichment
3. Build interactive heatmap view (division x impact)
4. Export final structured file
5. (Optional) Link to whitepaper Q\&A via embeddings

Let me know if you need prompt templates, dashboard code, or clause samples.
