Here is the complete, final `README.md` file code optimized for your repository. It includes all sections, clear explanations, the precise code architecture, and deployment instructions in one single, copy-pasteable block.

---

```markdown
# 🩺 Med AI: Tri-Pipeline GraphRAG Architecture Suite

A high-scale Systems Audit and Benchmarking Platform engineered to evaluate clinical intelligence retrieval across massive database matrices.

---

## 📌 Project Overview

The core objective of this project is to solve a fundamental enterprise AI bottleneck: **Scaling clinical data retrieval without inducing massive latency or context bloat.** Med AI benchmarks **three distinct retrieval architectures** side-by-side against a massive, 33-column production database engine (`med_ai_100M_engine.csv`) containing millions of clinical tokens. The entire ecosystem is monitored via a multi-threaded, asynchronous telemetry controller that tracks execution latency, token counts, commercial costs, and factual consistency in real-time.

---

## 🛠️ The Three Competing Retrieval Engines


```

```
                [ USER CLINICAL QUERY ]
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │  Pipeline 1  │   │  Pipeline 2  │   │  Pipeline 3  │
 │ Brute Force  │   │  Vector RAG  │   │   GraphRAG   │
 └──────────────┘   └──────────────┘   └──────────────┘

```

```

### 🔴 Pipeline 1: Raw Brute-Force Framework (`P1TK.PY`)
* **Mechanism:** Loads the entire 33-column medical database into local RAM using Pandas, running a linear keyword search across all parameters to construct a raw prompt context.
* **Bottleneck:** High context bloat, extreme token hemorrhage, and linear scanning delays that scale poorly with data volume.

### 🟡 Pipeline 2: Vector Semantic Indexing (`P2TK.py`)
* **Mechanism:** Utilizes `SentenceTransformer("all-MiniLM-L6-v2")` to vectorize dense text fields, querying a persistent local `ChromaDB` cluster to retrieve relevant semantic chunks.
* **Bottleneck:** Suffers from **Context Loss**. High-dimensional semantic slicing frequently drops strict relational linkages between multi-tiered conditions, symptoms, and staging variants.

### 🔵 Pipeline 3: Enterprise GraphRAG Topology (`P3TK.PY`)
* **Mechanism:** Queries localized structural database topology, mapping entities to explicit network coordinates (**Vertices** like *Diseases* or *SymptomClusters*, and **Edges** like *MANIFESTS_AS* or *MANAGED_BY*).
* **Result:** Delivers precise clinical context directly to the inference engine with sub-second latency and minimal token consumption.

---

## 📊 Automated Cross-Examination & Performance Reporting

To maintain a rigorous, production-grade benchmark, the project features a decoupled testing suite:

* **`A FINAL DASHBOARD.PY`:** The primary multi-window interface that launches synchronized query sweeps across all three concurrent threads, rendering live telemetry displays.
* **`Benchmark.py`:** An isolated console benchmarking engine that sweeps across 5 complex target nodes (*Asthma, Chronic Inflammatory Disorder, Moderate Acinic Cell Carcinoma, Refractory Warthin Tumor, and Esophageal Cancer*). It tracks metrics in both **USD ($)** and **INR (₹)** while generating an automated tabular performance summary.
* **Simulation Metrics:** Incorporates deterministic jitter matrices tracking **BERTScore F1 alignment** and **LLM-as-a-Judge Relevance Rankings** to prevent fixed-value bias.

---

## 📦 Repository Structure

Your workspace deployment contains the following file assets:
```text
├── A FINAL DASHBOARD.PY    # Unified Tri-Pipeline Cross-Examiner GUI
├── Benchmark.py            # Console-based Aligned Performance Auditor
├── P1TK.PY                 # Pipeline 1 Interface (RAM Brute-Force)
├── P2TK.py                 # Pipeline 2 Interface (ChromaDB Vector RAG)
├── P3TK.PY                 # Pipeline 3 Interface (GraphRAG Topology Center)
├── REQUIREMENTS.TXT        # Global Python System Dependencies
└── DRIVE LINK TO DATASET   # Access reference to underlying med_ai_100M_engine.csv

```

---

## ⚙️ Quick Start & Environment Setup

### Prerequisites

* Windows Subsystem for Linux (WSL) / macOS Terminal
* Docker Desktop running locally
* Python 3.10+ installed

### 1. Initialize the Graph Database Server Container

Open your system terminal and pull down the engine node. We map the network gateway interfaces directly onto your local workspace environment ports:

```bash
docker run -d -p 8000:9000 -p 14240:14240 -p 9002:9002 --name tigergraph_medic_ai -v ~/data:/home/tigergraph/mydata -t [docker.tigergraph.com/tigergraph-dev:latest](https://docker.tigergraph.com/tigergraph-dev:latest)

```

### 2. Configure System Dependencies

Install the required application layers, analytical engines, and vector environments listed in your repository configuration:

```bash
pip install -r REQUIREMENTS.TXT

```

### 3. Establish Runtime Environment Security Variables

To protect credentials, the platform pulls authorization tokens dynamically from your environment layer. Inject your API key into your session before bootstrapping the system:

```bash
# For Bash / Linux / macOS / WSL
export GEMINI_API_KEY="your_api_key_here"

# For Windows Command Prompt
set GEMINI_API_KEY=your_api_key_here

# For Windows PowerShell
$env:GEMINI_API_KEY="your_api_key_here"

```

### 4. Execute the Systems Audit

You can now spin up the master graphical radar array or trigger the terminal batch benchmark runner natively:

```bash
# Launch the Unified Cross-Examiner GUI
python "A FINAL DASHBOARD.PY"

# Or run the strict automated performance ledger
python Benchmark.py

```

```
***

```
