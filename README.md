# 🧠 Simple RAG System - Document Query Assistant

This project implements a lightweight **Retrieval-Augmented Generation (RAG)** system using **Streamlit**, **FAISS**, and **LLMs** like OpenAI and Ollama. It allows users to **upload PDF, TXT, or DOCX documents**, embed them into a searchable vector database, and query them using natural language. The model then returns **context-aware answers** based on retrieved document chunks.

---

## 🚀 Features

- 📄 Upload **PDF**, **TXT**, or **DOCX** files
- 🔍 Split content into **semantic chunks** with overlapping context
- 📦 Vector store backed by **FAISS** for efficient similarity search
- 🔗 Embeddings from **OpenAI**, **Nomic**, or **Chroma**
- 🧠 LLM-based question answering via **OpenAI (GPT-4)** or **Ollama (LLaMA3)**
- 🧵 Streamlit UI with **model selection**, caching, and source highlighting

---
