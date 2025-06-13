import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from transformers import pipeline
from odf.opendocument import load
from odf.text import P
import fitz  # PyMuPDF
import docx
import os
import threading

# Supported models
models_info = {
    "Fast & Light (DistilBERT)": "distilbert-base-uncased-distilled-squad",
    "Balanced (MiniLM)": "deepset/minilm-uncased-squad2",
    "Accurate (BERT Large)": "bert-large-uncased-whole-word-masking-finetuned-squad",
    "Multilingual (XLM-RoBERTa)": "deepset/xlm-roberta-base-squad2"
}

def chunk_text(text, max_tokens=400):
    words = text.split()
    return [' '.join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

def read_txt_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def read_odt_file(path):
    doc = load(path)
    text = ""
    for p in doc.getElementsByType(P):
        if p.firstChild:
            text += p.firstChild.data + "\n"
    return text

def read_pdf_file(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def read_docx_file(path):
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def browse_file():
    path = filedialog.askopenfilename(filetypes=[("Supported files", "*.txt *.pdf *.odt *.docx")])
    if path:
        entry_file.delete(0, tk.END)
        entry_file.insert(0, path)

def set_status(text):
    status_label.config(text=text)
    root.update_idletasks()

def set_progress(value):
    progress_bar['value'] = value
    root.update_idletasks()

def process_file_thread():
    try:
        file_path = entry_file.get()
        question = entry_question.get()
        model_key = model_choice.get()

        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid file.")
            set_status("Idle")
            set_progress(0)
            return
        if not question.strip():
            messagebox.showwarning("Warning", "Please enter a question.")
            set_status("Idle")
            set_progress(0)
            return

        ext = os.path.splitext(file_path)[1].lower()

        set_status("Reading file...")
        set_progress(10)

        if ext == ".txt":
            text = read_txt_file(file_path)
        elif ext == ".odt":
            text = read_odt_file(file_path)
        elif ext == ".pdf":
            text = read_pdf_file(file_path)
        elif ext == ".docx":
            text = read_docx_file(file_path)
        else:
            messagebox.showerror("Unsupported", "Supported: .txt, .odt, .pdf, .docx")
            set_status("Idle")
            set_progress(0)
            return

        set_status(f"Loading model: {model_key} ...")
        set_progress(30)
        qa = pipeline("question-answering", model=models_info[model_key])

        set_status("Answering question...")
        set_progress(60)

        chunks = chunk_text(text)
        best_answer = ""
        best_score = 0

        for i, chunk in enumerate(chunks):
            result = qa(question=question, context=chunk)
            if result['score'] > best_score:
                best_score = result['score']
                best_answer = result['answer']
            # update progress for chunks dynamically (from 60 to 90)
            chunk_progress = 60 + int(30 * (i + 1) / len(chunks))
            set_progress(chunk_progress)

        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, best_answer)

        with open("answer.txt", "w", encoding="utf-8") as f:
            f.write(best_answer)

        set_status("Done! Answer saved to answer.txt")
        set_progress(100)
    except Exception as e:
        messagebox.showerror("Error", str(e))
        set_status("Idle")
        set_progress(0)

def process_file():
    # run the processing in a separate thread so GUI stays responsive
    threading.Thread(target=process_file_thread, daemon=True).start()

# ----------------- GUI -----------------
root = tk.Tk()
root.title("AI Document Q&A Tool with Progress")

# File selection
tk.Label(root, text="Select File:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
entry_file = tk.Entry(root, width=50)
entry_file.grid(row=0, column=1, padx=5)
tk.Button(root, text="Browse", command=browse_file).grid(row=0, column=2, padx=5)

# Question input
tk.Label(root, text="Your Question:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
entry_question = tk.Entry(root, width=50)
entry_question.grid(row=1, column=1, padx=5)

# Model dropdown
tk.Label(root, text="Choose Model:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
model_choice = tk.StringVar()
model_dropdown = tk.OptionMenu(root, model_choice, *models_info.keys())
model_choice.set("Fast & Light (DistilBERT)")
model_dropdown.grid(row=2, column=1, sticky='w', padx=5)

# Get answer button
tk.Button(root, text="Get Answer", command=process_file, bg="lightblue").grid(row=2, column=2, padx=5)

# Progress bar and status label
progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.grid(row=3, column=1, padx=5, pady=5, sticky='w')

status_label = tk.Label(root, text="Idle")
status_label.grid(row=3, column=0, padx=5, pady=5, sticky='e')

# Output box
tk.Label(root, text="Answer:").grid(row=4, column=0, sticky='ne', padx=5, pady=5)
output_text = tk.Text(root, height=10, width=70)
output_text.grid(row=4, column=1, columnspan=2, padx=5, pady=5)

root.mainloop()

