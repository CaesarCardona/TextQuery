import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from transformers import pipeline
import os
import threading

# Supported models
models_info = {
    "Fast & Light (DistilBERT)": "distilbert-base-uncased-distilled-squad",
    "Balanced (MiniLM)": "deepset/minilm-uncased-squad2",
    "Accurate (BERT Large)": "bert-large-uncased-whole-word-masking-finetuned-squad",
    "Multilingual (XLM-RoBERTa)": "deepset/xlm-roberta-base-squad2"
}

# Global state
loaded_text = None
qa_model = None
file_loaded = False

def chunk_text(text, max_tokens=400):
    words = text.split()
    return [' '.join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

def read_txt_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def browse_file():
    global file_loaded, loaded_text
    path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not path:
        return
    entry_file.delete(0, tk.END)
    entry_file.insert(0, path)

    set_status("Reading file...")
    root.update()

    try:
        loaded_text = read_txt_file(path)
        file_loaded = True
        set_status("File loaded successfully.")
    except Exception as e:
        messagebox.showerror("Error", str(e))
        set_status("Idle")

def load_model():
    global qa_model
    model_key = model_choice.get()
    set_status(f"Loading model: {model_key} ...")
    root.update()
    qa_model = pipeline("question-answering", model=models_info[model_key])
    set_status("Model ready!")

def set_status(text):
    status_label.config(text=text)
    root.update_idletasks()

def ask_question_thread():
    global qa_model, loaded_text
    question = entry_question.get().strip()

    if not file_loaded:
        messagebox.showerror("Error", "Please load a file first.")
        return
    if not question:
        messagebox.showwarning("Warning", "Please enter a question.")
        return
    if qa_model is None:
        load_model()

    threading.Thread(target=answer_question, args=(question,), daemon=True).start()

def answer_question(question):
    global qa_model, loaded_text
    set_status("Answering...")
    progress_bar['value'] = 0
    root.update_idletasks()

    chunks = chunk_text(loaded_text)
    best_answer = ""
    best_score = 0

    for i, chunk in enumerate(chunks):
        result = qa_model(question=question, context=chunk)
        if result['score'] > best_score:
            best_score = result['score']
            best_answer = result['answer']
        progress_bar['value'] = int((i + 1) / len(chunks) * 100)
        root.update_idletasks()

    output_text.insert(tk.END, f"\n🧠 You: {question}\n🤖 AI: {best_answer}\n")
    output_text.see(tk.END)
    entry_question.delete(0, tk.END)
    set_status("Ready")

# ----------------- GUI -----------------
root = tk.Tk()
root.title("AI TXT Q&A Chat")

# File selection
tk.Label(root, text="Select File:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
entry_file = tk.Entry(root, width=50)
entry_file.grid(row=0, column=1, padx=5)
tk.Button(root, text="Browse", command=browse_file).grid(row=0, column=2, padx=5)

# Model dropdown
tk.Label(root, text="Model:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
model_choice = tk.StringVar()
model_choice.set("Fast & Light (DistilBERT)")
tk.OptionMenu(root, model_choice, *models_info.keys()).grid(row=1, column=1, sticky='w', padx=5)

# Question input
tk.Label(root, text="Ask a question:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
entry_question = tk.Entry(root, width=50)
entry_question.grid(row=2, column=1, padx=5)
tk.Button(root, text="Ask", command=ask_question_thread, bg="lightblue").grid(row=2, column=2, padx=5)

# Progress bar
progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.grid(row=3, column=1, padx=5, pady=5, sticky='w')

status_label = tk.Label(root, text="Idle")
status_label.grid(row=3, column=0, padx=5, pady=5, sticky='e')

# Output chat box
tk.Label(root, text="Chat:").grid(row=4, column=0, sticky='ne', padx=5, pady=5)
output_text = tk.Text(root, height=20, width=80, wrap='word')
output_text.grid(row=4, column=1, columnspan=2, padx=5, pady=5)

root.mainloop()

