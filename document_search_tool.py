#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 19:30:00 2025
Updated on Tue Jun 10 21:00:00 2025
@author: marek (w/ Gemini)

This module provides the DocumentSearchTool class. It now functions as an
"Agentic Document Researcher" that retrieves multiple document chunks, uses an
LLM to synthesize an answer, and identifies the best page image to display.
"""
import os
import base64
import requests
import json
import re
from typing import Dict, Optional, List

try:
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    print("FATAL: Langchain libraries not found. The DocumentSearchTool cannot function.")
    exit(1)

class DocumentSearchTool:
    """An agentic tool to search a document knowledge base, synthesize findings, and retrieve linked images."""

    def __init__(self, db_path: str = "Database/pdf_knowledge_base", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print("--- Initializing Document Search Tool ---")
        if not os.path.isdir(db_path):
            raise FileNotFoundError(f"Chroma DB path not found at '{db_path}'. Please run ingest_papers.py first.")

        # --- NEW: Configuration for internal LLM calls ---
        self.ollama_endpoint = "http://localhost:11434/api/generate"
        self.synthesis_model = "gemma3:12b-it-qat"

        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': 'cpu'})
            self.db = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
            print("--- Document Search Tool Initialized Successfully ---")
        except Exception as e:
            print(f"\n--- FATAL: Document Search Tool failed to initialize ---")
            raise e
            
    def _call_llm(self, prompt: str) -> str:
        """A generic internal helper to call the Ollama LLM."""
        payload = {"model": self.synthesis_model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}
        try:
            response = requests.post(self.ollama_endpoint, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.RequestException as e:
            print(f"  -> LLM call failed: {e}")
            return ""

    def _encode_image(self, image_path: str) -> Optional[str]:
        # ... (This helper function is unchanged)
        if not image_path or image_path == "N/A" or not os.path.exists(image_path):
            print(f"  -> Image path not found or invalid: {image_path}")
            return None
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"  -> Error encoding image {image_path}: {e}")
            return None

    # --- MODIFIED: The search method is now a multi-step agentic process ---
    def search(self, query: str) -> Dict[str, Optional[str]]:
        """
        Performs an agentic search: retrieves multiple chunks, uses an LLM
        to synthesize an answer, and returns the answer with the most relevant image.
        """
        print(f"DOC_SEARCH: Starting agentic search for: '{query}'")
        try:
            # Step 1: Perform a broader search to get multiple candidate chunks
            print("  -> Step 1: Retrieving top 5 candidate chunks from vector store...")
            candidate_docs = self.db.similarity_search(query, k=5)

            if not candidate_docs:
                return {'text_context': 'No relevant documents found.', 'image_base64': None}

            # Step 2: Assemble the context package for the synthesis agent
            print("  -> Step 2: Assembling evidence package for analysis...")
            evidence_package = ""
            for doc in candidate_docs:
                metadata = doc.metadata
                source_pdf = metadata.get('source_pdf', 'Unknown')
                page_num = metadata.get('page_number', 'N/A')
                evidence_package += f"Source: {source_pdf}, Page: {page_num}\nContent: {doc.page_content}\n\n---\n\n"

            # Step 3: Call the Synthesis Agent LLM
            print("  -> Step 3: Calling Synthesis LLM to analyze evidence...")
            synthesis_prompt = f"""
            You are a research assistant. Your task is to answer the user's query based ONLY on the provided text chunks from a document.

            **User's Query:** "{query}"

            **Provided Text Chunks:**
            {evidence_package}

            **Instructions:**
            1.  Read all the provided text chunks carefully.
            2.  Synthesize a single, comprehensive answer to the user's query.
            3.  Identify the single most relevant page number from the chunks that best supports your answer or contains a key figure/table related to the query.
            4.  Respond with ONLY a valid JSON object in the format:
                {{"answer": "Your synthesized answer here.", "most_relevant_page": <page_number_as_integer>}}
            """
            
            llm_response_str = self._call_llm(synthesis_prompt)
            
            # Step 4: Parse the response and retrieve the final image
            print("  -> Step 4: Parsing response and retrieving final image...")
            text_context = "Could not synthesize an answer from the retrieved documents."
            image_base64 = None
            
            try:
                json_match = re.search(r'\{.*\}', llm_response_str, re.DOTALL)
                if not json_match: raise ValueError("No JSON object found in response.")
                
                synthesis_result = json.loads(json_match.group(0))
                text_context = synthesis_result.get("answer", text_context)
                relevant_page_num = synthesis_result.get("most_relevant_page")

                if relevant_page_num:
                    print(f"  -> Synthesizer identified page {relevant_page_num} as most relevant.")
                    # Find the document corresponding to the most relevant page
                    for doc in candidate_docs:
                        if doc.metadata.get('page_number') == relevant_page_num:
                            image_path = doc.metadata.get('image_path')
                            image_base64 = self._encode_image(image_path)
                            if image_base64:
                                print(f"  -> Successfully encoded linked image: {image_path}")
                            break # Stop after finding the first match
            
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"  -> Warning: Could not parse synthesizer response JSON: {e}. Returning text only.")
                text_context = llm_response_str # Return the raw text if JSON fails

            return {'text_context': text_context, 'image_base64': image_base64}

        except Exception as e:
            print(f"An unexpected error occurred during document search: {e}")
            return {'text_context': f"An error occurred: {e}", 'image_base64': None}


if __name__ == '__main__':
    # ... (The demo block remains the same, but the output should now be much better)
    print("--- Running Document Search Tool Demo (Agentic Version) ---")
    try:
        doc_search_tool = DocumentSearchTool()
        test_query = "What does the STA-1 method do and how does it compare to other watermarks?"
        result = doc_search_tool.search(test_query)
        print("\n--- Search Result ---")
        print("Synthesized Answer:")
        print(result['text_context'])
        print("\nMost Relevant Image Retrieved:")
        print(f"{'Yes' if result['image_base64'] else 'No'}")
    except Exception as e:
        import traceback
        print(f"\nDemo failed during setup or execution. Error: {e}")
        traceback.print_exc()