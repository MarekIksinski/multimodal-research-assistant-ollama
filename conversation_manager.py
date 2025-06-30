#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 8 15:50:00 2025
Updated on Tue Jun 10 23:00:00 2025

@author: marek (w/ Gemini)

This module provides a ConversationManager class to handle persistent storage
of conversations using SQLite (for text) and ChromaDB (for vectors).
Knowledge Graph logic has been moved to knowledge_graph_tool.py.
"""
import sqlite3
import os
import datetime
from typing import List, Dict, Optional

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
except ImportError:
    print("FATAL: Langchain libraries not found. The ConversationManager cannot function.")
    exit(1)

class ConversationManager:
    """Manages the storage and retrieval of conversation history."""

    def __init__(self, base_dir: str = "Database", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"--- Initializing Conversation Manager ---")
        os.makedirs(base_dir, exist_ok=True)
        self.sqlite_path = os.path.join(base_dir, "conversations.sqlite")
        self.chroma_path = os.path.join(base_dir, "conversation_vectors")

        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': 'cpu'})
            self.db_conn = sqlite3.connect(self.sqlite_path)
            self.vector_store = Chroma(persist_directory=self.chroma_path, embedding_function=self.embeddings)
            self._create_sqlite_tables()
            print("--- Conversation Manager Initialized Successfully ---")
        except Exception as e:
            print(f"\n--- FATAL: Conversation Manager failed to initialize ---")
            raise e

    def _create_sqlite_tables(self):
        cursor = self.db_conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY, start_time TEXT, name TEXT, session_summary TEXT);")
        cursor.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, conversation_id INTEGER, timestamp TEXT, role TEXT, content TEXT, FOREIGN KEY (conversation_id) REFERENCES conversations (id));")
        self.db_conn.commit()

    def add_turn_to_history(self, conversation_id: int, user_content: str, assistant_content: str):
        """Adds a conversation turn to SQLite and ChromaDB."""
        try:
            cursor = self.db_conn.cursor()
            now = datetime.datetime.now().isoformat()
            cursor.execute("INSERT INTO messages (conversation_id, timestamp, role, content) VALUES (?, ?, ?, ?)", (conversation_id, now, 'user', user_content))
            user_message_id = cursor.lastrowid
            cursor.execute("INSERT INTO messages (conversation_id, timestamp, role, content) VALUES (?, ?, ?, ?)", (conversation_id, now, 'assistant', assistant_content))
            assistant_message_id = cursor.lastrowid
            
            combined_content = f"User: {user_content}\n\nAssistant: {assistant_content}"
            metadata = {"conversation_id": conversation_id, "user_message_id": user_message_id, "assistant_message_id": assistant_message_id, "timestamp": now}
            chroma_id = f"turn_{user_message_id}_{assistant_message_id}"
            
            self.vector_store.add_texts(texts=[combined_content], metadatas=[metadata], ids=[chroma_id])
            self.db_conn.commit()
            print(f"Successfully added turn (User: {user_message_id}, Assistant: {assistant_message_id}) to conversation {conversation_id}.")
        except Exception as e:
            print(f"Error adding turn to history: {e}")
            self.db_conn.rollback()
            
    # Other methods like start_new_conversation, list_conversations, etc. are unchanged
    def start_new_conversation(self, name: Optional[str] = None) -> int:
        start_time = datetime.datetime.now().isoformat()
        cursor = self.db_conn.cursor()
        cursor.execute("INSERT INTO conversations (start_time, name) VALUES (?, ?)", (start_time, name))
        self.db_conn.commit()
        return cursor.lastrowid

    def list_conversations(self) -> List[Dict]:
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT id, start_time, name, session_summary FROM conversations ORDER BY start_time DESC")
        rows = cursor.fetchall()
        return [{"id": r[0], "start_time": r[1], "name": r[2], "summary_exists": bool(r[3])} for r in rows]

    def update_conversation_summary(self, conversation_id: int, summary: str):
        cursor = self.db_conn.cursor()
        cursor.execute("UPDATE conversations SET session_summary = ? WHERE id = ?", (summary, conversation_id))
        self.db_conn.commit()

    def get_conversation_summary(self, conversation_id: int) -> Optional[str]:
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT session_summary FROM conversations WHERE id = ?", (conversation_id,))
        result = cursor.fetchone()
        return result[0] if result and result[0] else None