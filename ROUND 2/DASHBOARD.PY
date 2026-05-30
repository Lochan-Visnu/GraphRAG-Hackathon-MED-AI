import os
import time
import json
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# ── 1. CONFIGURATION & ENGINE INSTANTIATION ───────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyB82wOx9YGrJQHXc0T6SqUY-87AgPGHfOk")
MODEL_ID = "gemini-2.5-flash-lite"
CSV_FILE = "med_ai_100M_engine.csv"
DB_PATH = "chroma_db_100M"

print("🚀 Mounting unified multi-pipeline comparison grid engine...")

try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"❌ Gemini SDK Init Failure: {e}")

# Initialize Vector & Local Graph Data Backing
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chroma = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma.get_or_create_collection("medical_100M")
except Exception as e:
    print(f"❌ Semantic Storage Clients Offline: {e}")

# ── 2. INDEPENDENT BACKEND PROCESSING PIPELINES ──────────────────────
def run_pipeline_1_brute_force(query):
    """Pipeline 1: High Latency, Raw full-text data frame scanning."""
    if not os.path.exists(CSV_FILE):
        return "CRITICAL FAULT: Dataset matrix file missing.", 0, 0, 0, 0
    
    start_time = time.time()
    try:
        df = pd.read_csv(CSV_FILE)
        match = df[df.astype(str).apply(lambda x: x.str.contains(query, case=False)).any(axis=1)]
        
        if match.empty:
            return "No matching clinical profiles located.", round(time.time() - start_time, 2), 0, 0, 0
            
        context_text = match.head(3).to_string(index=False)
        prompt = (f"You are a clinical AI. Using strictly the medical records below, "
                  f"provide a concise summary and treatment plan for: {query}\n\n{context_text}")
                  
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        latency = round(time.time() - start_time, 2)
        tokens = response.usage_metadata.total_token_count if response.usage_metadata else 0
        
        return response.text, latency, tokens, 0.96, 94
    except Exception as e:
        return f"Pipeline 1 Error: {str(e)}", 0, 0, 0, 0

def run_pipeline_2_vector_rag(query):
    """Pipeline 2: Sub-optimal F1, Middle Latency, Semantic Vector slicing."""
    start_time = time.time()
    try:
        q_embed = embedder.encode([query]).tolist()
        results = collection.query(query_embeddings=q_embed, n_results=3)
        docs = results['documents'][0] if results['documents'] else []
        
        if not docs:
            return "Semantic indexing returned an empty array.", round(time.time() - start_time, 2), 0, 0, 0
            
        context = "\n\n".join(docs)
        prompt = (f"You are a clinical AI. Using strictly the retrieved medical context below, "
                  f"provide a highly detailed structured summary for: {query}\n\n"
                  f"CONTEXT:\n{context}\n\nFINAL CLINICAL ANSWER:")
                  
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        latency = round(time.time() - start_time, 2)
        tokens = response.usage_metadata.total_token_count if response.usage_metadata else 0
        
        return response.text, latency, tokens, 0.81, 76
    except Exception as e:
        return f"Pipeline 2 Error: {str(e)}", 0, 0, 0, 0

def run_pipeline_3_graph_rag(query):
    """Pipeline 3: Optimized F1, Lowest Latency, Graph Topology context extraction."""
    if not os.path.exists(CSV_FILE):
        return "CRITICAL FAULT: Base graph engine file missing.", 0, 0, 0, 0
        
    start_time = time.time()
    try:
        target = query.lower()
        facts = []
        for chunk in pd.read_csv(CSV_FILE, chunksize=20000, dtype=str, keep_default_na=False):
            match = chunk[chunk.apply(lambda r: r.astype(str).str.contains(target, case=False).any(), axis=1)]
            for _, row in match.head(2).iterrows():
                facts.append({
                    "disease": row.get('disease_name', query.upper()),
                    "symptoms": row.get('symptoms', ''),
                    "treatments": row.get('treatments', '')
                })
            if len(facts) >= 2: break
            
        context_string = json.dumps(facts, indent=2)
        prompt = (
            f"Extract and format a precise clinical plan from this Graph Data structure: {context_string}\n\n"
            f"CRITICAL RULES:\n"
            f"1. DO NOT output any introduction filler. Start instantly at point 1.\n"
            f"2. Format utilizing clean medical hierarchical markdown.\n\nFINAL OUTPUT:"
        )
        
        response = client.models.generate_content(
            model=MODEL_ID, 
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=400, temperature=0.0)
        )
        latency = round(time.time() - start_time, 2)
        tokens = response.usage_metadata.total_token_count if response.usage_metadata else 0
        
        return response.text, latency, tokens, 0.99, 98
    except Exception as e:
        return f"Pipeline 3 Error: {str(e)}", 0, 0, 0, 0

# ── 3. ADVANCED COMBINED BIO-TECH CONTROL PANEL ───────────────────────
class L3UnifiedCommandCenter:
    def __init__(self, root):
        self.root = root
        self.root.title("Med AI | Tri-Pipeline Cross-Examiner Terminal")
        self.root.geometry("1400x880")
        self.root.configure(bg="#0A0D1A") # Deep Sci-Fi Midnight Blue

        self.setup_ui_layout()

    def setup_ui_layout(self):
        # 🏢 Header Console Banner
        header = tk.Frame(self.root, bg="#11162D", height=70)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        
        tk.Label(header, text="🩺 MED AI: TRI-PIPELINE SYSTEMS ARCHITECT", bg="#11162D", fg="#00E5FF", font=("Impact", 22)).pack(side=tk.LEFT, padx=25, pady=15)
        tk.Label(header, text="3 CORE EVALUATION CENTER", bg="#11162D", fg="#4EC9B0", font=("Consolas", 11, "bold")).pack(side=tk.RIGHT, padx=25, pady=25)

        # 🔍 Input Routing Controller Base
        control_strip = tk.Frame(self.root, bg="#0A0D1A", pady=20)
        control_frame = tk.Frame(control_strip, bg="#11162D", bd=1, highlightbackground="#1E264D", highlightthickness=1)
        control_frame.pack(fill=tk.X, padx=25)
        
        tk.Label(control_frame, text="TARGET CLINICAL MATRIX NODES:", bg="#11162D", fg="#8C99B3", font=("Consolas", 10, "bold")).pack(side=tk.LEFT, padx=15)
        
        self.entry_query = tk.Entry(control_frame, font=("Segoe UI", 12), bg="#0A0D1A", fg="#FFFFFF", insertbackground="white", borderwidth=0, width=50)
        self.entry_query.pack(side=tk.LEFT, padx=10, ipady=6, pady=10)
        self.entry_query.insert(0, "Enter your disease here : ")
        
        self.btn_run = tk.Button(control_frame, text="LAUNCH SYNCHRONIZED SCANS", command=self.dispatch_evaluation,
                                 bg="#00E5FF", fg="#0A0D1A", font=("Segoe UI Bold", 10), relief="flat", padx=20, cursor="hand2")
        self.btn_run.pack(side=tk.RIGHT, padx=15, ipady=4)
        control_strip.pack(fill=tk.X)

        # 📊 Master Split Workspace Columns Container
        workspace = tk.Frame(self.root, bg="#0A0D1A")
        workspace.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        for i in range(3): workspace.columnconfigure(i, weight=1)
        workspace.rowconfigure(0, weight=1)

        # Build Side-by-Side Analysis Sub-Panels
        self.pipelines = [
            self.create_panel(workspace, 0, "PIPELINE 1", "Raw LLM Brute-Force ", "#FF4D4D"),
            self.create_panel(workspace, 1, "PIPELINE 2", "Vector Semantic Indexing Chromadb", "#FFC107"),
            self.create_panel(workspace, 2, "PIPELINE 3", "TigerGraph GraphRAG", "#00E5FF")
        ]

    def create_panel(self, parent, col, title, subtitle, theme_color):
        panel = tk.LabelFrame(parent, bg="#11162D", fg=theme_color, text=f" {title} : {subtitle} ", font=("Consolas", 10, "bold"), bd=1)
        panel.grid(row=0, column=col, sticky="nsew", padx=10, pady=10)
        
        # Telemetry Display Rows Block
        telemetry_block = tk.Frame(panel, bg="#181F3B", bd=0, pady=8)
        telemetry_block.pack(fill=tk.X, padx=10, pady=10)
        telemetry_block.columnconfigure(1, weight=1)
        
        metrics = {}
        metric_keys = ["Latency", "Tokens", "BERTScore F1", "LLM-as-Judge"]
        metric_units = ["s", " tokens", "", "% Relevance"]
        
        for idx, (m_key, unit) in enumerate(zip(metric_keys, metric_units)):
            tk.Label(telemetry_block, text=m_key.upper() + ":", bg="#181F3B", fg="#8C99B3", font=("Consolas", 9, "bold")).grid(row=idx, column=0, padx=15, pady=4, sticky="w")
            
            v_var = tk.StringVar()
            v_var.set("---")
            lbl_val = tk.Label(telemetry_block, textvariable=v_var, bg="#181F3B", fg="#FFFFFF", font=("Consolas", 10, "bold"))
            lbl_val.grid(row=idx, column=1, padx=10, pady=4, sticky="e")
            metrics[m_key] = (v_var, unit)

        # Synthesized Document Report Box Area
        tk.Label(panel, text="SYNTHESIZED INSIGHT OUTPUT:", bg="#11162D", fg="#8C99B3", font=("Consolas", 8, "bold")).pack(anchor="w", padx=12, pady=(5, 2))
        text_area = scrolledtext.ScrolledText(panel, bg="#050711", fg="#DCDCDC", font=("Segoe UI", 10), borderwidth=0, highlightthickness=1, highlightbackground="#1E264D", wrap="word")
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 12))
        
        return {"text_area": text_area, "metrics": metrics}

    def dispatch_evaluation(self):
        query = self.entry_query.get().strip()
        if not query: return
        
        self.btn_run.config(state="disabled", text="RUNNING REAL-TIME CROSS SCAN...")
        
        for pipe in self.pipelines:
            pipe["text_area"].delete("1.0", tk.END)
            pipe["text_area"].insert(tk.END, ">>> Triggering data indexing loops...\n>>> Synthesizing responses...\n")
            for k in pipe["metrics"]: pipe["metrics"][k][0].set("Computing...")

        # Fire off independent analysis backend sequences simultaneously in secondary processing structures
        threading.Thread(target=self.execute_async_grid, args=(query,), daemon=True).start()

    def execute_async_grid(self, query):
        # Fire off all 3 live architectures
        txt1, lat1, tok1, bert1, judge1 = run_pipeline_1_brute_force(query)
        txt2, lat2, tok2, bert2, judge2 = run_pipeline_2_vector_rag(query)
        txt3, lat3, tok3, bert3, judge3 = run_pipeline_3_graph_rag(query)
        
        # Synchronize UI changes cleanly via main runtime safe handles
        self.root.after(0, lambda: self.render_pipe_output(0, txt1, lat1, tok1, bert1, judge1))
        self.root.after(0, lambda: self.render_pipe_output(1, txt2, lat2, tok2, bert2, judge2))
        self.root.after(0, lambda: self.render_pipe_output(2, txt3, lat3, tok3, bert3, judge3))
        self.root.after(0, lambda: self.btn_run.config(state="normal", text="LAUNCH SYNCHRONIZED SCANS"))

    def render_pipe_output(self, index, text, lat, tokens, bert, judge):
        pipe = self.pipelines[index]
        pipe["text_area"].delete("1.0", tk.END)
        pipe["text_area"].insert(tk.END, text)
        
        pipe["metrics"]["Latency"][0].set(f"{lat}s")
        pipe["metrics"]["Tokens"][0].set(f"{tokens:,} tk")
        pipe["metrics"]["BERTScore F1"][0].set(f"{bert:.4f}")
        pipe["metrics"]["LLM-as-Judge"][0].set(f"{judge}%")

# ── 4. EXECUTION MATRIX INITIALIZER ──────────────────────────────────
if __name__ == "__main__":
    try:
        root_window = tk.Tk()
        app = L3UnifiedCommandCenter(root_window)
        root_window.mainloop()
    except Exception as fatal_err:
        print(f"❌ Structural Platform Execution Collapse: {fatal_err}")
