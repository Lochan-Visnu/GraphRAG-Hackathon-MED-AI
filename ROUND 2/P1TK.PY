import pandas as pd
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
from google import genai
import threading
import os

# ── 1. CONFIGURATION (SECURED ENVIRONMENT KEYS) ───────────────────────
# API key is pulled safely via environment variables. Do not hardcode strings here.
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
CSV_FILE = "med_ai_100M_engine.csv" 
MODEL_ID = "gemini-2.5-flash-lite" 

# Initialize client using secured credentials object reference
client = genai.Client(api_key=API_KEY)

def get_cost_bracket(total_tokens):
    """Abstracts micro-cents into a production cost bracket for the dashboard."""
    if total_tokens == 0: return "-"
    if total_tokens < 5000: return "LOW"
    if total_tokens < 20000: return "MEDIUM"
    return "HIGH"

# ── 2. ADVANCED TELEMETRY UI CLASS ────────────────────────────────────
class MedAIPipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Med AI | Pipeline 1 Auditor (Base LLM)")
        self.root.geometry("1000x750")
        self.root.configure(bg="#0F111A") 

        self.setup_styles()
        self.setup_ui()

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
        tk.Label(nav, text=f"3 CORE | {MODEL_ID.upper()}", bg="#1A1C27", fg="#4EC9B0", font=("Consolas", 10)).pack(side=tk.RIGHT, padx=20)

        # --- Input Section ---
        input_container = tk.Frame(self.root, bg="#0F111A", pady=30)
        input_container.pack()
        
        self.query_entry = tk.Entry(input_container, width=55, font=("Segoe UI", 12), bg="#1A1C27", fg="#DCDCDC", 
                                    insertbackground="white", borderwidth=0, highlightthickness=1, highlightbackground="#3E4452")
        self.query_entry.pack(side=tk.LEFT, padx=10, ipady=8)
        self.query_entry.insert(0, "Enter clinical query (e.g., Asthma therapeutic protocols)")
        
        self.query_entry.bind("<FocusIn>", lambda e: self.query_entry.delete(0, 'end') if "Enter clinical query" in self.query_entry.get() else None)
        self.query_entry.bind("<Return>", lambda e: self.start_analysis_thread())

        self.search_btn = tk.Button(input_container, text="START RAW SCAN", command=self.start_analysis_thread, 
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

        tk.Label(output_frame, text="LIVE SYSTEM TELEMETRY", bg="#0F111A", fg="#808080", font=("Segoe UI Bold", 8)).pack(anchor="w")
        self.console = scrolledtext.ScrolledText(output_frame, bg="#050505", fg="#9CDCFE", font=("Consolas", 11), 
                                                borderwidth=0, highlightthickness=1, highlightbackground="#333")
        self.console.pack(fill=tk.BOTH, expand=True)
        self.write_console("System Ready. Awaiting user input...", "#808080")

    def create_stat_card(self, parent, title, value, color, col):
        card = tk.Frame(parent, bg="#1A1C27", padx=30, pady=15, highlightthickness=1, highlightbackground="#333")
        card.grid(row=0, column=col, padx=10)
        card.pack_propagate(False) if hasattr(card, 'pack_propagate') else None
        tk.Label(card, text=title, bg="#1A1C27", fg="#808080", font=("Segoe UI", 8, "bold")).pack()
        lbl = tk.Label(card, text=value, bg="#1A1C27", fg=color, font=("Segoe UI Semibold", 18))
        lbl.pack()
        return lbl

    def write_console(self, msg, color="#9CDCFE"):
        self.console.insert(tk.END, f"> {msg}\n")
        self.console.see(tk.END)

    def start_analysis_thread(self):
        query = self.query_entry.get().strip()
        if not query or "Enter clinical query" in query: return
        
        self.search_btn.config(state="disabled", bg="#333", text="LOADING RAM...")
        threading.Thread(target=self.run_logic, args=(query,), daemon=True).start()

    def run_logic(self, query):
        self.console.delete(1.0, tk.END)
        self.write_console(f"Initializing Auditor Session for target: [ {query} ]", "#4EC9B0")
        
        start_time = time.time()
        
        try:
            # ── PHASE 1: RAW FULL-FILE LOAD (NO CHUNKING) ──
            self.write_console(f"Allocating Memory: Loading entire 100M Token Database into RAM...", "#CE9178")
            self.write_console("Warning: Massive memory consumption anticipated. Standby...", "#F44747")
            
            df = pd.read_csv(CSV_FILE)
            
            self.write_console(f"Database Loaded. Scanning {len(df):,} rows for clinical matches...", "#808080")
            match = df[df.astype(str).apply(lambda x: x.str.contains(query, case=False)).any(axis=1)]
            
            search_end = time.time()
            search_latency = round(search_end - start_time, 2)

            if match.empty:
                self.write_console(f"Alert: Zero matches found for '{query}' in dataset.", "#F44747")
                self.reset_ui()
                return

            found_rows = match.head(5)
            context_text = found_rows.to_string(index=False)
            self.write_console(f"Retrieval Complete. Found relevant records in {search_latency}s.", "#DCDCAA")

            # ── PHASE 2: GENERATION (GEMINI NEW SDK) ──
            self.write_console(f"Model Dispatch: Forwarding extracted context to {MODEL_ID}...")
            
            prompt = (f"You are a clinical AI. Using strictly the medical records below, "
                      f"provide a concise summary and treatment plan for: {query}\n\n{context_text}")
            
            response = client.models.generate_content(
                model=MODEL_ID, 
                contents=prompt
            )
            
            total_end = time.time()
            total_latency = round(total_end - start_time, 2)

            # ── PHASE 3: TELEMETRY ──
            in_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
            out_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
            total_tokens = response.usage_metadata.total_token_count if response.usage_metadata else 0
            
            cost_bracket = get_cost_bracket(total_tokens)
            self.root.after(0, self.update_final_ui, response.text, total_latency, total_tokens, cost_bracket)

        except FileNotFoundError:
            self.write_console(f"CRITICAL ERROR: '{CSV_FILE}' not found.", "#F44747")
            self.reset_ui()
        except MemoryError:
            self.write_console(f"CRITICAL SYSTEM FAILURE: Out of Memory (RAM).", "#F44747")
            self.write_console(f"The 100M token dataset is too massive for a standard linear load.", "#F44747")
            self.reset_ui()
        except Exception as e:
            self.write_console(f"CRITICAL API/SYSTEM ERROR: {str(e)}", "#F44747")
            self.reset_ui()

    def update_final_ui(self, ai_text, latency, tokens, cost_bracket):
        self.write_console("\n" + "═"*40, "#4EC9B0")
        self.write_console("── ANALYSIS REPORT GENERATED ──", "#4EC9B0")
        self.write_console("═"*40 + "\n", "#4EC9B0")
        
        self.console.insert(tk.END, f"{ai_text}\n\n")
        self.lat_box.config(text=f"{latency}s")
        self.tok_box.config(text=f"{tokens:,}")
        
        cost_color = "#CE9178" if cost_bracket == "HIGH" else "#4EC9B0"
        self.cst_box.config(text=f"{cost_bracket}", fg=cost_color)
        self.reset_ui()

    def reset_ui(self):
        self.search_btn.config(state="normal", bg="#007ACC", text="START RAW SCAN")

if __name__ == "__main__":
    root = tk.Tk()
    app = MedAIPipelineGUI(root)
    root.mainloop()
