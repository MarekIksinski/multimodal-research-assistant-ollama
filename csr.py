#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 8 15:01:00 2025

@author: marek (adapted by Gemini)

This module defines the ConversationalSearchAgent class. Its purpose is to
act as the engine for a 'conversation_search' tool. It encapsulates the
agentic RAG logic to retrieve relevant turns from a conversation history
vector database and use an LLM to extract and summarize the key information.
"""
import requests
import json
import os
from typing import List, Dict, Optional

# --- RAG Imports ---
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
except ImportError:
    print("FATAL: Langchain libraries not found. The CSR module cannot function.")
    print("Install with: pip install langchain langchain-community sentence-transformers langchain-huggingface")
    exit(1)

# ==============================================================================
# --- ConversationalSearchAgent Class Definition ---
# ==============================================================================

class ConversationalSearchAgent:
    """
    An agent that performs a RAG process on a conversation history vector DB.
    It retrieves relevant conversation turns and uses an LLM to extract
    and summarize the information relevant to a user's query.
    """
    def __init__(
        self,
        ollama_endpoint: str = "http://localhost:11434/api/chat",
        extraction_model_name: str = 'gemma3:12b-it-qat',
        model_options: Optional[Dict] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        conversation_db_path: str = './conversation_db', # Path for the conversation history DB
        num_retrieved_docs: int = 5,
        max_context_chars_per_doc: int = 1500,
        extraction_prompt: Optional[str] = None
    ):
        """
        Initializes the ConversationalSearchAgent.

        Args:
            ollama_endpoint: URL for the Ollama /api/chat endpoint.
            extraction_model_name: The Ollama model to use for the extraction step.
            model_options: Dictionary of options for the Ollama model.
            embedding_model: Name of the sentence-transformer model for embeddings.
            conversation_db_path: Path to the Chroma DB for conversation history.
            num_retrieved_docs: Number of past conversation turns to retrieve.
            max_context_chars_per_doc: Max characters per retrieved doc context.
            extraction_prompt: System prompt for the extraction LLM.
        """
        self.ollama_endpoint = ollama_endpoint
        self.extraction_model_name = extraction_model_name
        self.model_options = model_options if model_options is not None else {
            'temperature': 0.3, 'num_ctx': 8192, 'top_p': 0.8
        }
        self.conversation_db_path = conversation_db_path
        self.num_retrieved_docs = num_retrieved_docs
        self.max_context_chars_per_doc = max_context_chars_per_doc

        # REFACTORED: The agent now only needs an extraction prompt.
        self.extraction_prompt = extraction_prompt if extraction_prompt is not None else self._get_default_extraction_prompt()

        self.embeddings = self._initialize_embeddings(embedding_model)
        self.conversation_db = self._initialize_db()

    def _get_default_extraction_prompt(self) -> str:
        return """
You are an information extraction assistant. Carefully read the provided CONTEXT from a past conversation and the USER'S CURRENT QUERY.
Your task is to extract only the specific sentences or pieces of information from the CONTEXT that are directly relevant to answering the USER'S CURRENT QUERY.
Output only the extracted relevant information, formatted clearly and concisely.
If no relevant information is found in the provided context, output the exact phrase: No relevant information found.
"""

    def _initialize_embeddings(self, embedding_model_name: str) -> Optional[HuggingFaceEmbeddings]:
        """Initializes and returns the embeddings model."""
        print("CSR_Agent: Initializing Embeddings model...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={'device': 'cpu'} # Consider 'cuda' if GPU available
            )
            print(f"CSR_Agent: Embeddings model '{embedding_model_name}' loaded.")
            return embeddings
        except Exception as e:
            print(f"CSR_Agent: FATAL - Error initializing Embeddings model: {e}")
            return None

    def _initialize_db(self) -> Optional[Chroma]:
        """Initializes and returns the Chroma DB connection."""
        if not self.embeddings:
            print("CSR_Agent: Cannot initialize DB because embeddings failed to load.")
            return None

        print(f"CSR_Agent: Initializing Conversation DB connection ('{self.conversation_db_path}')...")
        try:
            if not os.path.isdir(self.conversation_db_path):
                 raise FileNotFoundError(f"Conversation DB directory not found: {self.conversation_db_path}")

            db = Chroma(
                persist_directory=self.conversation_db_path,
                embedding_function=self.embeddings
            )
            print("CSR_Agent: Conversation DB connection established.")
            return db
        except Exception as e:
            print(f"CSR_Agent: Error initializing Conversation DB: {e}")
            return None

    def _call_ollama_api(self, payload: dict) -> dict:
        """Internal helper to send requests to the Ollama API."""
        try:
            response = requests.post(self.ollama_endpoint, json=payload, timeout=180)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"CSR_Agent API Error: {e}")
            raise ConnectionError(f"Failed to connect to Ollama API: {e}") from e

    def _retrieve_context(self, query: str) -> str:
        """Retrieves context from the conversation Chroma DB."""
        if not self.conversation_db:
            print("CSR_Agent: Context retrieval skipped, DB not available.")
            return ""

        print(f"CSR_Agent: Searching conversation history for query: '{query}'")
        try:
            retrieved_docs = self.conversation_db.similarity_search(query, k=self.num_retrieved_docs)
            if not retrieved_docs:
                return ""

            print(f"CSR_Agent: Retrieved {len(retrieved_docs)} candidate conversation turns.")
            context_parts = []
            for i, doc in enumerate(retrieved_docs):
                doc_text = doc.page_content
                # Assuming metadata might contain 'role' and 'timestamp'
                source_info = f"Turn {doc.metadata.get('turn_id', i+1)}, Role: {doc.metadata.get('role', 'unknown')}"
                truncated_text = doc_text[:self.max_context_chars_per_doc]
                context_parts.append(f"--- Context from: ({source_info}) ---\n{truncated_text}\n")
            return "\n".join(context_parts)
        except Exception as e:
            print(f"CSR_Agent: Error during context retrieval: {e}")
            return ""

    def _extract_info_with_llm(self, context: str, query: str) -> str:
        """Uses an LLM to extract relevant info from the retrieved context."""
        if not context:
            print("CSR_Agent: Skipping extraction step: No context retrieved.")
            return ""

        print("CSR_Agent: Extracting relevant info via LLM from retrieved context...")
        try:
            extraction_messages = [
                {'role': 'system', 'content': self.extraction_prompt},
                {'role': 'user', 'content': f"CONTEXT FROM PAST CONVERSATION:\n{context}\n---\nUSER'S CURRENT QUERY: {query}"}
            ]
            extraction_payload = {
                "model": self.extraction_model_name,
                "messages": extraction_messages,
                "options": self.model_options,
                "stream": False
            }

            extraction_response = self._call_ollama_api(extraction_payload)
            extracted_info = extraction_response.get('message', {}).get('content', '').strip()

            # Check if the model explicitly states no info found
            if "no relevant information found" in extracted_info.lower() or len(extracted_info) < 10:
                print("CSR_Agent: Extraction LLM indicated no relevant info found.")
                return ""
            
            print(f"CSR_Agent: Extraction complete. Returning refined context.")
            return extracted_info

        except (ConnectionError, Exception) as e:
            print(f"CSR_Agent: Error during extraction LLM call: {e}")
            return ""

    def search(self, query: str) -> str:
        """
        The main public method for this agent. Executes the full RAG pipeline.

        Args:
            query: The search query, likely derived from the user's latest input.

        Returns:
            A string containing the summarized, relevant context, or an empty
            string if no relevant context was found or an error occurred.
        """
        # Step 1: Retrieve candidate turns from the conversation DB
        retrieved_context = self._retrieve_context(query)

        # Step 2: Use LLM to extract and refine the retrieved context
        refined_context = self._extract_info_with_llm(retrieved_context, query)

        return refined_context