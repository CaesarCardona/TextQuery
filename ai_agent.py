import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import requests
import subprocess

# ----------------- LangChain -----------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ----------------- MongoDB -----------------
from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/")
db = client['ai_qa_chat']
collection = db['queries']

def save_to_db(q, a):
    collection.insert_one({"question": q, "answer": a})

# ----------------- Globals -----------------
qa_chain = None
file_loaded = False

# ----------------- UI Helpers -----------------
def set_status(text):
    status_label.config(text=text)
    root.update_idletasks()

# ----------------- File -----------------
def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ----------------- Ollama Auto Setup -----------------
def ensure_ollama_model(model="mistral"):
    try:
        requests.get("http://localhost:11434")
    except:
        messagebox.showerror("Error", "Ollama is not running. Please start it.")
        return False

    res = requests.get("http://localhost:11434/api/tags").json()
    installed = [m["name"] for m in res.get("models", [])]

    if model not in installed:
        set_status(f"Downloading {model}...")
        root.update()
        subprocess.run(["ollama", "pull", model])

    return True

# ----------------- Build RAG -----------------
def build_rag(text):
    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = splitter.create_documents([text])

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Ensure model exists
    if not ensure_ollama_model("mistral"):
        return None

    llm = Ollama(model="mistral")

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
""")

    # Format docs
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 🔥 Modern RAG pipeline
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# ----------------- File Load -----------------
def browse_file():
    global qa_chain, file_loaded

    path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not path:
        return

    entry_file.delete(0, tk.END)
    entry_file.insert(0, path)

    set_status("Processing file...")
    root.update()

    try:
        text = read_txt(path)
        qa_chain = build_rag(text)

        if qa_chain:
            file_loaded = True
            set_status("Ready (Local RAG ✔)")
        else:
            set_status("Error")

    except Exception as e:
        messagebox.showerror("Error", str(e))
        set_status("Idle")

# ----------------- Ask -----------------
def ask_thread():
    q = entry_question.get().strip()

    if not file_loaded:
        messagebox.showerror("Error", "Load a file first")
        return

    if not q:
        return

    threading.Thread(target=answer, args=(q,), daemon=True).start()

def answer(q):
    set_status("Thinking...")
    progress_bar['value'] = 50
    root.update_idletasks()

    try:
        # 🔥 NEW CALL
        a = qa_chain.invoke(q)

        output_text.insert(tk.END, f"\n🧠 You: {q}\n🤖 AI: {a}\n")
        output_text.see(tk.END)

        save_to_db(q, a)

    except Exception as e:
        messagebox.showerror("Error", str(e))

    entry_question.delete(0, tk.END)
    progress_bar['value'] = 100
    set_status("Ready")

# ----------------- GUI -----------------
root = tk.Tk()
root.title("Local RAG Chat (Auto Models)")

# File
tk.Label(root, text="File:").grid(row=0, column=0, padx=5, pady=5)
entry_file = tk.Entry(root, width=50)
entry_file.grid(row=0, column=1)
tk.Button(root, text="Browse", command=browse_file).grid(row=0, column=2)

# Question
tk.Label(root, text="Question:").grid(row=1, column=0)
entry_question = tk.Entry(root, width=50)
entry_question.grid(row=1, column=1)
tk.Button(root, text="Ask", command=ask_thread, bg="lightblue").grid(row=1, column=2)

# Progress
progress_bar = ttk.Progressbar(root, length=400)
progress_bar.grid(row=2, column=1)

status_label = tk.Label(root, text="Idle")
status_label.grid(row=2, column=0)

# Output
output_text = tk.Text(root, height=20, width=80)
output_text.grid(row=3, column=0, columnspan=3)

root.mainloop()
