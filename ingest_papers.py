#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 18:00:00 2025
@author: marek (w/ Gemini)

This is an advanced ingestion pipeline for scientific papers. It performs:
1.  PDF-to-Image conversion for each page.
2.  Abstract extraction using an LLM.
3.  Knowledge Graph population from the abstract via the KnowledgeGraphTool.
4.  Vector store creation, linking text chunks to their page image.
"""
import requests
import os
import json
from tinydb import TinyDB, Query
from datetime import datetime

# --- NEW: Import for PDF-to-Image conversion ---
from pdf2image import convert_from_path

# --- Our Existing and New Tools ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from kgr import KnowledgeGraphTool # We will use our KG tool here

# --- Configuration ---
BASE_DIR = "Database"
RECORDS_DB_PATH = os.path.join(BASE_DIR, "ingestion_records.json")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "pdf_knowledge_base") # Vector store for document text
PDF_IMAGE_DIR = os.path.join(BASE_DIR, "pdf_images") # To store page images
PDF_FILES_LIST = './pdf-files.txt'
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
EXTRACTION_MODEL = "gemma3:12b-it-qat"

# --- One-time Initializations ---
print("--- Initializing Ingestion Pipeline ---")
os.makedirs(PDF_IMAGE_DIR, exist_ok=True)
records_db = TinyDB(RECORDS_DB_PATH)
ProcessedFile = Query()
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL, model_kwargs={'device': 'cpu'})
chroma_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
# Initialize our KG Tool, which will manage the graph file itself
kg_tool = KnowledgeGraphTool(base_dir=BASE_DIR)
print("--- Initialization Complete ---")


def extract_abstract_with_llm(text: str) -> str:
    """Uses an LLM to find and extract the abstract from the first page text."""
    print("\t-> Attempting to extract abstract with LLM...")
    prompt = f"""
    You are a text extraction assistant. From the following text, which is from the first page of a scientific paper, please extract and return ONLY the full text of the 'Abstract' section. Do not include the word "Abstract". If you cannot find an abstract, return an empty string.

    Text:
    {text}
    """
    payload = {"model": EXTRACTION_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}}
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=90)
        response.raise_for_status()
        abstract = response.json().get("response", "").strip()
        if abstract:
            print("\t-> Abstract extracted successfully.")
        else:
            print("\t-> No abstract found by LLM.")
        return abstract
    except requests.RequestException as e:
        print(f"\t-> LLM call for abstract extraction failed: {e}")
        return ""

def process_pdf(pdf_path: str):
    """
    The main processing function for a single PDF, now with robust
    image path handling.
    """
    if records_db.contains(ProcessedFile.pdf_path == pdf_path):
        print(f"Skipping already processed file: {pdf_path}")
        return

    print(f"--- Processing: {pdf_path} ---")
    pdf_filename = os.path.basename(pdf_path)
    
    # 1. Load the document text
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
    except Exception as e:
        print(f"\tError loading PDF: {e}")
        records_db.insert({'pdf_path': pdf_path, 'status': 'error_loading', 'timestamp': datetime.now().isoformat()})
        return

    # 2. Extract Abstract and Build Knowledge Graph
    if pages:
        first_page_text = pages[0].page_content
        abstract_text = extract_abstract_with_llm(first_page_text)
        if abstract_text:
            kg_tool.update_graph_from_text(abstract_text)

    # --- MODIFIED: More robust PDF-to-Image conversion and path management ---
    print(f"\t-> Converting PDF to images...")
    image_paths = [] # To store the actual paths of saved images
    image_output_dir = os.path.join(PDF_IMAGE_DIR, os.path.splitext(pdf_filename)[0])
    os.makedirs(image_output_dir, exist_ok=True)
    
    try:
        # Convert PDF to a list of image objects
        images = convert_from_path(pdf_path, fmt='jpeg')
        
        # Loop through the image objects, save them with predictable names, and store the paths
        for i, image in enumerate(images):
            page_num = i + 1
            # Create a clear and predictable filename
            image_path = os.path.join(image_output_dir, f"page_{page_num}.jpg")
            image.save(image_path, 'JPEG')
            image_paths.append(image_path)
        
        print(f"\t-> Successfully converted and saved {len(image_paths)} pages.")
    except Exception as e:
        print(f"\t-> PDF to image conversion failed: {e}. Multimodal features will be disabled for this document.")
    # --- END OF MODIFICATION ---

    # 4. Chunk and Embed Page by Page, linking to images
    print("\t-> Processing pages for vector store...")
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    for i, page in enumerate(pages):
        page_num = i + 1
        page_text = page.page_content
        chunks = text_splitter.create_documents([page_text])
        
        # Get the corresponding image path from our list (if it exists)
        # The index `i` will match because both `pages` and `image_paths` are 0-indexed lists
        linked_image_path = image_paths[i] if i < len(image_paths) else "N/A"

        for chunk in chunks:
            chunk.metadata['source_pdf'] = pdf_filename
            chunk.metadata['page_number'] = page_num
            chunk.metadata['image_path'] = linked_image_path
        
        all_chunks.extend(chunks)

    if all_chunks:
        print(f"\t-> Adding {len(all_chunks)} text chunks to ChromaDB...")
        chroma_db.add_documents(all_chunks)
        print("\t-> Chunks added successfully.")

    # 5. Record that this file has been processed
    records_db.insert({'pdf_path': pdf_path, 'timestamp': datetime.now().isoformat(), 'status': 'processed'})
    print(f"--- Finished processing: {pdf_path} ---\n")


def run_ingestion():
    """Main function to read the list of PDFs and process them."""
    if not os.path.exists(PDF_FILES_LIST):
        print(f"Error: PDF list file not found at '{PDF_FILES_LIST}'")
        print("Please create this file and add the full paths to your PDFs, one per line.")
        return

    with open(PDF_FILES_LIST, 'r') as f:
        pdf_paths = [line.strip() for line in f if line.strip()]

    print(f"Found {len(pdf_paths)} PDFs to process.")
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            process_pdf(pdf_path)
        else:
            print(f"Warning: File not found, skipping: {pdf_path}")
            
    # Persist the ChromaDB changes once at the very end
    print("\nPersisting all changes to ChromaDB...")
    chroma_db.persist()
    print("Ingestion process complete.")


if __name__ == "__main__":
    run_ingestion()