# Multi-Agent Conversational Debate System  

## Project Overview

This project implements a **multi-agent debate system** where three distinct personas — **Commander**, **Rationalist**, and **Dramatist** — engage in structured debates to collaboratively reason about complex queries. Their discussion concludes with a **Synthesizer Agent**, which merges all viewpoints into a single, unified insight.

The goal is to simulate realistic collaborative reasoning between agents with different cognitive biases and conversational styles. Each agent retrieves evidence from its **persona-specific knowledge base**, debates in multiple rounds, and contributes to the final synthesis.

The system is built using:
- **LangGraph** — for orchestration  
- **FAISS** — for retrieval  
- **Sentence-Transformers (SBERT)** — for embeddings  
- **Streamlit** — for user interaction  


## Personas & Discovery Process

The three personas were designed from patterns discovered in movie dialogues using data-driven persona discovery.

### Data Source
- **Dataset:** [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)  
- Contains extracted dialogues from films (`data/raw/movie_data`)  
- Each line includes the **character**, **movie**, and **dialogue text**

  
### Persona Discovery Workflow
1. **Preprocessing**
   - Lowercasing, punctuation removal, token filtering  
   - Removed non-dialogue noise and duplicates  
   - Filtered sentences between 5 and 220 characters

2. **Topic Modeling & Clustering**
   - Used **SentenceTransformer (all-MiniLM-L6-v2)** embeddings  
   - Clustered utterances by semantic similarity  
   - Manually labeled clusters based on tone and reasoning patterns

3. **Final Personas**

| Persona | Description | Traits |
|----------|--------------|--------|
| 🧭 **Commander** | Tactical and action-oriented | Decisive, structured, direct |
| 🧠 **Rationalist** | Analytical, logic-based thinker | Data-driven, hypothesis-testing |
| 🎭 **Dramatist** | Emotionally expressive | Reflective, metaphorical, humanistic |

4. **Synthesizer Agent**
   - Combines all three perspectives  
   - Produces a balanced, human-readable unified insight  

To regenerate persona clusters and `persona_analysis.json`:
```bash
python src/persona_discovery.py
```


## Collaboration Architecture

The system uses **LangGraph** to model agent interactions as a **state machine** of connected nodes.

| Phase | Description | Node Function |
|--------|--------------|----------------|
| **Init** | Initializes agents, state, and dialogue | `init_state_node` |
| **Round 1** | Each persona gives their independent viewpoint | `round1_node` |
| **Dialogue** | Agents converse & reference each other | `dialogue_node` |
| **Challenge** | Cross-critiques and rebuttals | `challenges_node` |
| **Synthesis** | Unified summary of insights | `synthesis_node` |

The `router` function manages transitions between phases based on conversation state.  
`GraphState` tracks queries, dialogue history, and agent responses.


## Rebuilding Data Artifacts (FAISS + Persona Meta)

We intentionally do **not** commit large artifacts. Rebuild locally:

1. Place a dialog dataset under `data/raw/movie_data/` (e.g., *Cornell Movie-Dialogs Corpus*).  
2. Preprocess & discover personas:
   ```bash
   PYTHONPATH=. python src/preprocessing.py
   PYTHONPATH=. python src/persona_discovery.py
   ```
3. Build persona indices:
   ```bash
   PYTHONPATH=. python src/build_indices.py
   ```

This generates:
```
data/processed/personas/<agent>/
  ├─ <agent>.meta.jsonl
  └─ <agent>.faiss
```


## ⚙️ Setup & Running Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/eashita11/vectorial-multiagent.git
cd vectorial-multiagent
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# or
.venv\Scripts\activate      # Windows
```

### 3️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

### 4️⃣ Start Ollama & Run App
Start the Ollama Llama3 server locally:
```bash
brew install ollama          # macOS
ollama serve
ollama pull llama3
```

Then launch Streamlit:
```bash
streamlit run app.py
```


## Tests

All test files are located in `/tests/`.

**Quick subset:**
```bash
PYTHONPATH=. pytest -q -k "retriever or citations"
```

**Full suite (can be slow on CPU):**
```bash
PYTHONPATH=. pytest -q --maxfail=1
```

> **Note:** FAISS and model initialization can slow tests down.


## Dataset & Preprocessing

Processed dataset files are large, so they are not included in the repository.  
To recreate them:

```bash
python src/preprocessing.py
python src/persona_discovery.py
```

You may alternatively use an open-source dataset such as the *Cornell Movie Dialogs Corpus*.


## Usage Guide

1. Launch the Streamlit app.  
2. Ask any question (e.g., “How can I rebuild trust in a failing team?”).  
3. Observe:
   - **Round 1:** Individual viewpoints  
   - **Dialogue:** Agents build on each other  
   - **Challenges:** Cross-critiques  
   - **Synthesis:** Unified insight  

### Showcase Prompts
- “We missed our Q3 retention target; CS says it’s product bugs, PM says onboarding. What should we do this sprint to stabilize churn?”  
- “Marketing wants to launch in 4 weeks; infra warns cost blow-up at 3× traffic. How do we decide, and what’s the minimum we ship safely?”


## Citation & Attribution System

Each agent uses a **retriever** backed by **FAISS** and **SBERT embeddings**:
- Each persona retrieves snippets aligned with its reasoning style.
- Citations appear in the UI (e.g., `🔗 HAIG · line L421488 · movie m150`).
- The **Synthesizer Agent** only reuses citations provided by personas — it never invents new ones.


## User Interface

A **Streamlit app (`app.py`)** provides a chat-based interface:

- Type your question into the input box.  
- Watch color-coded persona responses:
  - 🧭 **Commander** → Blue border  
  - 🧠 **Rationalist** → Green border  
  - 🎭 **Dramatist** → Purple border  
  - 🧩 **Synthesis** → Teal border  
- Smooth typing animation simulates a live debate.


## Repository Structure
```
├── app.py
├── data/
│   ├── raw/
│   └── processed/
│       ├── persona_analysis.json
│       └── personas/
├── requirements.txt
├── src/
│   ├── agents/
│   ├── graph/
│   ├── preprocessing.py
│   ├── persona_discovery.py
│   ├── retriever.py
│   ├── orchestrator.py
│   └── llm.py
└── tests/
```


## Limitations & Future Work

| Area | Current Limitation | Future Improvement |
|------|--------------------|--------------------|
| Dialogue Depth | Some agents dominate or repeat | Add memory balancing & attention control |
| Persona Contrast | Commander and Rationalist occasionally overlap | Refine prompts for stronger stylistic contrast |
| Speed | Local LLM inference is slow | Enable async calls or caching |
| Retrieval | Static FAISS indices | Add dynamic persona-specific retraining |


## Demo

Below is a glimpse of the Multi-Agent Debate in action:

| Stage | Screenshot |
|--------|-------------|
| 🗣️ Round 1 — Individual Perspectives | ![Round1](assets/round1.png) |
| 💬 Dialogue — Agents Build on Each Other | ![Dialogue](assets/dialogue.png) |
| ⚖️ Challenge Round — Rebuttals | ![Challenge](assets/challenge.png) |
| 🧩 Consensus Summary — Unified Insight | ![Synthesis](assets/synthesis.png) |

### Typing Animation Preview
![Typing Demo](assets/typing.gif)


## Credits

Developed by **Eashita Dhillon**  
