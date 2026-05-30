import time
import os
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# ── 1. CONFIGURATION (SECURED ENVIRONMENT KEYS) ───────────────────────
# API key is pulled safely via environment variables. Do not hardcode strings here.
API_KEY  = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
MODEL_ID = "gemini-2.5-flash-lite"

# Target Data Paths
CSV_FILE = "med_ai_100M_engine.csv"
DB_PATH  = "chroma_db_100M"

# ── INIT ENGINES ───────────────────────────────────────────────────
print("🚀 Initializing Vector Database Environment...")
client     = genai.Client(api_key=API_KEY)
embedder   = SentenceTransformer("all-MiniLM-L6-v2")
chroma     = chromadb.PersistentClient(path=DB_PATH)
collection = chroma.get_or_create_collection("medical_100M")

def get_cost_bracket(total_tokens):
    """Abstracts micro-cents into a production cost bracket for the dashboard."""
    if total_tokens == 0: return "-"
    if total_tokens < 5000: return "LOW"
    if total_tokens < 15000: return "MEDIUM"
    return "HIGH"

# ── ADVANCED TELEMETRY UI CLASS ───────────────────────────────────────
class MedAIPipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Med AI | Pipeline 2 Auditor (Vector RAG)")
        self.root.geometry("1000x800")
        self.root.configure(bg="#0F111A")

        self.setup_styles()
        self.setup_ui()
        
        # Smart Indexing Check
        if collection.count() == 0:
            threading.Thread(target=self.run_csv_indexing, daemon=True).start()
        else:
            self.write_console(f"Vector Database Online. Ready — {collection.count():,} semantic profiles loaded.", "#DCDCAA")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#0F111A")
        style.configure("Card.TFrame", background="#1A1C27", borderwidth=1)

    def setup_ui(self):
        # --- Top Navigation Bar ---
        nav = tk.Frame(self.root, bg="#1A1C27", height=60)
        nav.pack(fill=tk.X)
        tk.Label(nav, text="🩺 MED AI: SYSTEMS AUDITOR", bg="#1A1C27", fg="#569CD6", font=("Impact", 18)).pack(side=tk.LEFT, padx=20)
        tk.Label(nav, text=f"3 CORE | {MODEL_ID.upper()} + CHROMADB", bg="#1A1C27", fg="#4EC9B0", font=("Consolas", 10)).pack(side=tk.RIGHT, padx=20)

        # --- Input Section ---
        input_container = tk.Frame(self.root, bg="#0F111A", pady=30)
        input_container.pack()
        
        self.query_entry = tk.Entry(input_container, width=55, font=("Segoe UI", 12), bg="#1A1C27", fg="#DCDCDC", 
                                    insertbackground="white", borderwidth=0, highlightthickness=1, highlightbackground="#3E4452")
        self.query_entry.pack(side=tk.LEFT, padx=10, ipady=8)
        self.query_entry.insert(0, "Enter clinical query for semantic vector search...")
        
        self.query_entry.bind("<FocusIn>", lambda e: self.query_entry.delete(0, 'end') if "Enter clinical query" in self.query_entry.get() else None)
        self.query_entry.bind("<Return>", lambda e: self.start_analysis_thread())

        self.search_btn = tk.Button(input_container, text="START VECTOR SCAN", command=self.start_analysis_thread, 
                                   bg="#007ACC", fg="white", font=("Segoe UI Bold", 10), relief="flat", padx=25, pady=8, cursor="hand2")
        self.search_btn.pack(side=tk.LEFT)

        # --- Metrics Dashboard Grid ---
        metrics_container = tk.Frame(self.root, bg="#0F111A")
        metrics_container.pack(pady=10)

        self.lat_box = self.create_stat_card(metrics_container, "TOTAL LATENCY", "0.00s", "#CE9178", 0)
        self.tok_box = self.create_stat_card(metrics_container, "TOKENS PROCESSED", "0", "#4FC1FF", 1)
        self.cst_box = self.create_stat_card(metrics_container, "PRODUCTION COST", "-", "#DCDCAA", 2)

        # --- Live Terminal Output ---
        output_frame = tk.Frame(self.root, bg="#0F111A", padx=40)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=20)

        tk.Label(output_frame, text="LIVE SYSTEM TELEMETRY (PIPELINE 2)", bg="#0F111A", fg="#808080", font=("Segoe UI Bold", 8)).pack(anchor="w")
        self.console = scrolledtext.ScrolledText(output_frame, bg="#050505", fg="#9CDCFE", font=("Consolas", 11), 
                                                borderwidth=0, highlightthickness=1, highlightbackground="#333")
        self.console.pack(fill=tk.BOTH, expand=True)
        self.console.tag_config("white", foreground="#FFFFFF")
        self.write_console("System Ready. Awaiting user input...", "#808080")

    def create_stat_card(self, parent, title, value, color, col):
        card = tk.Frame(parent, bg="#1A1C27", padx=30, pady=15, highlightthickness=1, highlightbackground="#333")
        card.grid(row=0, column=col, padx=10)
        tk.Label(card, text=title, bg="#1A1C27", fg="#808080", font=("Segoe UI", 8, "bold")).pack()
        lbl = tk.Label(card, text=value, bg="#1A1C27", fg=color, font=("Segoe UI Semibold", 18))
        lbl.pack()
        return lbl

    def write_console(self, msg, color="#9CDCFE"):
        self.console.insert(tk.END, f"> {msg}\n")
        self.console.see(tk.END)

    def run_csv_indexing(self):
        self.search_btn.config(state="disabled", text="INDEXING DB...")
        self.write_console(f"📚 Initializing Vector Engine on {CSV_FILE}...", "#CE9178")
        self.write_console("Warning: Vectorizing 100M tokens locally is computationally expensive.", "#DCDCAA")
        self.write_console("Enabling Smart Sample limit (5,000 rows) for live demonstration purposes...", "#DCDCAA")
        
        if not os.path.exists(CSV_FILE):
            self.write_console(f"CRITICAL ERROR: File not found -> {CSV_FILE}", "#F44747")
            return
        
        try:
            processed = 0
            row_limit = 5000 
            
            for chunk in pd.read_csv(CSV_FILE, chunksize=1000):
                if processed >= row_limit:
                    break
                
                batch_texts = []
                for _, row in chunk.iterrows():
                    para = (f"Clinical Profile for {row.get('disease_name', 'Unknown')}: "
                            f"Symptoms include {row.get('symptoms', '')}. "
                            f"Standard treatments involve {row.get('treatments', '')} using {row.get('medicine_classes', '')}. "
                            f"Details: {row.get('disease_description', '')}")
                    batch_texts.append(para[:1000]) 
                
                embeddings = embedder.encode(batch_texts, show_progress_bar=False).tolist()
                ids = [f"csv_chunk_{processed + j}" for j in range(len(batch_texts))]
                
                try: collection.add(documents=batch_texts, embeddings=embeddings, ids=ids)
                except: pass
                
                processed += len(batch_texts)
                self.write_console(f"   Indexed {processed} clinical profiles into ChromaDB...", "#808080")
            
            self.write_console(f"✅ ChromaDB Indexed successfully: {collection.count():,} profiles available.", "#4EC9B0")
            self.reset_ui()
            
        except Exception as e:
            self.write_console(f"CRITICAL INDEXING ERROR: {str(e)}", "#F44747")
            self.reset_ui()

    def start_analysis_thread(self):
        query = self.query_entry.get().strip()
        if not query or "Enter clinical query" in query: return
        
        self.search_btn.config(state="disabled", bg="#333", text="QUERYING DB...")
        threading.Thread(target=self.run_rag_logic, args=(query,), daemon=True).start()

    def run_rag_logic(self, question):
        self.console.delete(1.0, tk.END)
        self.write_console(f"Initializing Auditor Session for target: [ {question} ]", "#4EC9B0")
        
        start_time = time.time()
        
        try:
            self.write_console("Vector Search: Engaging ChromaDB engine...", "#DCDCAA")
            
            q_embed = embedder.encode([question]).tolist()
            results = collection.query(query_embeddings=q_embed, n_results=3) 
            
            search_end = time.time()
            search_latency = round(search_end - start_time, 2)
            
            self.write_console(f"Retrieval Complete. Found {len(results['documents'][0])} semantic matches in {search_latency}s.", "#DCDCAA")
            self.write_console("🔎 Top Context Retrieved:", "#808080")
            
            for i, doc in enumerate(results["documents"][0][:3]): 
                self.write_console(f"  [{i+1}] {doc[:100]}...", "#808080")

            context = "\n\n".join(results["documents"][0])
            self.write_console(f"Model Dispatch: Forwarding extracted context to {MODEL_ID}...")
            
            prompt = (f"You are a top-tier clinical AI. Using strictly the retrieved medical context below, "
                      f"provide a highly detailed, structured summary and treatment plan for: {question}\n\n"
                      f"Format your response with clear headings and numbered/bulleted lists for readability, "
                      f"matching standard clinical treatment protocols.\n\n"
                      f"CONTEXT:\n{context}\n\n"
                      f"QUESTION: {question}\nFINAL ANSWER:")

            response = client.models.generate_content(
                model=MODEL_ID, 
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=1000, 
                    temperature=0.1
                )
            )
            
            total_end = time.time()
            total_latency = round(total_end - start_time, 2)

            in_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
            out_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
            total_tokens = response.usage_metadata.total_token_count if response.usage_metadata else 0
            
            cost_bracket = get_cost_bracket(total_tokens)
            self.root.after(0, self.update_final_ui, response.text, total_latency, total_tokens, cost_bracket)

        except Exception as e:
            self.write_console(f"CRITICAL API/SYSTEM ERROR: {str(e)}", "#F44747")
            self.reset_ui()

    def update_final_ui(self, ai_text, latency, tokens, cost_bracket):
        self.write_console("\n" + "═"*40, "#4EC9B0")
        self.write_console("── ANALYSIS REPORT GENERATED ──", "#4EC9B0")
        self.write_console("═"*40 + "\n", "#4EC9B0")
        
        self.console.insert(tk.END, f"{ai_text}\n\n", "white")
        self.lat_box.config(text=f"{latency}s")
        self.tok_box.config(text=f"{tokens:,}")
        
        cost_color = "#CE9178" if cost_bracket == "HIGH" else "#4EC9B0"
        self.cst_box.config(text=f"{cost_bracket}", fg=cost_color)
        self.reset_ui()

    def reset_ui(self):
        self.search_btn.config(state="normal", bg="#007ACC", text="START VECTOR SCAN")
        self.query_entry.delete(0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = MedAIPipelineGUI(root)
    root.mainloop()
