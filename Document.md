## 🧠 Regulatory Compliance Agentic Framework for Financial Institutions

### 🎯 Objective

Build a multi-agent system to analyze regulatory changes, extract obligations, assess impact across applications and teams, and track progress in a human-in-the-loop (HITL) compliant manner.

---

### 🔁 Regulation Processing Flow Diagram

![Regulatory Agent Flow Diagram](https://user-images.githubusercontent.com/placeholder/regulatory-agent-flow.png)

> *(Diagram shows the sequential and branched relationship between each agent, highlighting HITL checkpoints.)*

---

### 🔁 Regulation Processing Pipeline (Detailed)

1. **Regulation Ingestion Agent**

   * Parses PDF/text/email to extract metadata and regulation body.
   * Input: Regulation document (e.g., PDF)
   * Output:

     ```json
     {
       "title": "CDCTCD-2025",
       "effective_date": "2025-10-01",
       "text": "...raw regulation content..."
     }
     ```
   * HITL: ❌ Optional (manual correction if OCR fails)

2. **Obligation Extraction Agent**

   * Converts raw regulation into actionable obligations.
   * Generates unique `task_id`s.
   * Output:

     ```json
     [
       { "id": "OB-1", "text": "Encrypt all customer data at rest" },
       { "id": "OB-2", "text": "Obtain consent for cross-border transfers" }
     ]
     ```
   * HITL: ✅ Required for review/edit

3. **Impact Analysis Agent**

   * Maps each obligation to affected applications and teams.
   * Output:

     ```json
     {
       "apps": ["ClientVault", "CRM360"],
       "teams": ["Cloud Security", "Infrastructure"]
     }
     ```
   * HITL: ✅ Confirmation required

4. **Owner Assignment Agent**

   * Resolves and assigns task owners from org chart.
   * Output:

     ```json
     { "owner": "Ankit Jain" }
     ```
   * HITL: ✅ Optional override

5. **Timeline Planning Agent**

   * Suggests milestone timelines per obligation.
   * Output:

     ```json
     {
       "start_date": "2025-10-05",
       "end_date": "2025-11-30",
       "milestones": [
         { "name": "Design", "duration": "2w" },
         { "name": "Implementation", "duration": "4w" },
         { "name": "Validation", "duration": "2w" }
       ]
     }
     ```
   * HITL: ✅ Edit milestones/dates

6. **Execution Tracker Agent**

   * Tracks progress from Jira/Slack systems.
   * Output:

     ```json
     { "status": "in progress", "percent_complete": 60 }
     ```
   * HITL: ❌ Autonomous unless failure occurs

7. **Escalation Detection Agent**

   * Monitors overdue/missing updates; flags escalation.
   * Output:

     ```json
     { "action": "escalate", "reason": "Task OB-2 overdue by 6 days" }
     ```
   * HITL: ✅ Escalation requires approval

8. **Supervisor Agent**

   * Provides oversight, querying, overrides.
   * Output:

     ```json
     { "task_id": "OB-1", "status": "blocked" }
     ```
   * HITL: ✅ Human may pause/redirect flow

---

### 📚 Required Knowledge Sources

| Type                 | Source Example                    | Used By               |
| -------------------- | --------------------------------- | --------------------- |
| Regulation Docs      | FINMA\_CDCTCD-2025.pdf            | Ingestion, Extraction |
| App Registry         | ClientVault → Security            | Impact Analysis       |
| Team Mapping         | Cloud Security → Ankit Jain       | Owner Assignment      |
| Org Chart            | LDAP, CSV, HR systems             | Owner Assignment      |
| Historical Timelines | Past Jira projects, workload DB   | Timeline Planner      |
| Escalation Rules     | `{ "Ankit": "Rina" }`             | Escalation Agent      |
| Task Trackers        | Jira/Trello, Slack update monitor | Execution Tracker     |

---

### ✅ HITL Matrix

| Agent / Node          | Human in Loop? | Why Required                             |
| --------------------- | -------------- | ---------------------------------------- |
| Regulation Ingestion  | ❌ Optional     | Only if OCR/format issues                |
| Obligation Extraction | ✅ Yes          | AI may miss/mis-split clauses            |
| Impact Analysis       | ✅ Yes          | Domain context critical                  |
| Owner Assignment      | ✅ Yes          | Organizational exceptions possible       |
| Timeline Planning     | ✅ Yes          | Business calendar, priority input needed |
| Execution Tracker     | ❌ No           | Agent syncs directly                     |
| Escalation Detection  | ✅ Yes          | Sensitive to context                     |
| Supervisor Agent      | ✅ Yes          | Manual override control                  |

---

### 🚀 Example Use Case

**Input Regulation:**

> "Encrypt all customer data at rest using NIST standards. Obtain explicit consent for all cross-border transfers."

**Steps:**

* Extracted:

  * OB-1: Encrypt customer data
  * OB-2: Consent management for transfers
* Apps Affected:

  * OB-1 → ClientVault, CRM360
  * OB-2 → ConsentDB, WealthPortal
* Owners:

  * OB-1 → Ankit Jain
  * OB-2 → Priya Mehta
* Timelines:

  * OB-1: Oct 5 – Nov 30
  * OB-2: Oct 10 – Dec 15
* Execution:

  * OB-1 at 60%, OB-2 delayed → Escalation triggered

---

This document forms the architectural foundation for your regulatory compliance hackathon project.
