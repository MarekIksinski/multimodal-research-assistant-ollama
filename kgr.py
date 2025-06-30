#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 23:00:00 2025
Updated on Tue Jun 10 12:44:00 2025

@author: marek (w/ Gemini)

This module provides the KnowledgeGraphTool class.
It now uses a precise Subject-Predicate-Object (SPO) model to extract
high-quality facts and build a cleaner, more accurate knowledge graph.
"""
import os
import requests
import json
import re
import spacy
import networkx as nx
from typing import Optional, List

class KnowledgeGraphTool:
    """Manages all operations related to the conversational knowledge graph."""

    def __init__(self, base_dir: str = "Database"):
        print("--- Initializing Knowledge Graph Tool ---")
        os.makedirs(base_dir, exist_ok=True)
        self.graph_path = os.path.join(base_dir, "knowledge_graph.gpickle")

        self.ollama_endpoint = "http://localhost:11434/api/generate"
        self.extraction_model = "gemma3:12b-it-qat"

        try:
            self.nlp = self._load_spacy_model()
            self.graph = self._load_or_create_graph()
            print("--- Knowledge Graph Tool Initialized Successfully ---")
        except Exception as e:
            print(f"\n--- FATAL: Knowledge Graph Tool failed to initialize ---")
            raise e

    def _load_spacy_model(self):
        """Loads the spacy model, downloading if necessary."""
        print("KG_Tool: Initializing spacy model...")
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("KG_Tool: Spacy 'en_core_web_sm' model not found. Downloading...")
            spacy.cli.download("en_core_web_sm")
            return spacy.load("en_core_web_sm")

    def _load_or_create_graph(self) -> nx.Graph:
        """Loads the knowledge graph from file or creates a new one."""
        print("KG_Tool: Initializing Knowledge Graph...")
        if os.path.exists(self.graph_path):
            print(f"KG_Tool: Loading existing knowledge graph from '{self.graph_path}'")
            return nx.read_gpickle(self.graph_path)
        else:
            print("KG_Tool: Creating new knowledge graph.")
            return nx.Graph()

    def _call_llm(self, prompt: str) -> str:
        """A generic internal helper to call the Ollama LLM."""
        payload = {"model": self.extraction_model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}}
        try:
            response = requests.post(self.ollama_endpoint, json=payload, timeout=90)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.RequestException as e:
            print(f"KG_Tool: LLM call failed: {e}")
            return ""

    # --- MODIFIED: This is the new, more precise fact extraction method ---
    def _extract_facts_from_sentence(self, sentence: str):
        """
        Uses an LLM to extract Subject-Predicate-Object triples from a single sentence.
        """
        prompt = f"""
        You are a highly precise information extraction assistant. From the following sentence, extract all factual relationships as a structured (Subject, Predicate, Object) triple. The 'Predicate' should be a concise verb phrase that describes the relationship.

        - Extract only clear, direct relationships.
        - If no facts can be extracted, return an empty list `[]`.
        - Your entire response MUST be ONLY a valid JSON object containing a list of lists.
        - Example: [["Subject", "Predicate", "Object"], ["Another Subject", "Predicate", "Object"]]

        Sentence: "{sentence}"
        """
        response = self._call_llm(prompt)

        try:
            # Use regex to find the JSON list, robust against LLM chatter
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return # No valid list found

            triples = json.loads(json_match.group(0))
            
            # Validate that we got a list of lists with 3 elements each
            if isinstance(triples, list):
                for triple in triples:
                    if isinstance(triple, list) and len(triple) == 3:
                        subject, predicate, obj = (s.strip() for s in triple)
                        
                        # Skip adding if the exact same fact (edge with same label) already exists
                        if self.graph.has_edge(subject, obj) and self.graph[subject][obj].get('label') == predicate:
                            print(f"    - Skipping duplicate fact: ({subject}) -> [{predicate}] -> ({obj})")
                            continue
                        
                        self.graph.add_edge(subject, obj, label=predicate)
                        print(f"    + Added Fact: ({subject}) -> [{predicate}] -> ({obj})")

        except (json.JSONDecodeError, TypeError):
            print(f"  -> Could not parse SPO from sentence: '{sentence}'")
            return

    def update_graph_from_text(self, text: str):
        """
        Public method to orchestrate fact extraction for a blob of text.
        This now uses the more precise SPO extraction method.
        """
        print("KG_Tool: Updating Knowledge Graph with SPO extractor...")
        doc = self.nlp(text)
        for sentence in doc.sents:
            # We only process sentences that are reasonably long to contain facts
            if len(sentence.text.split()) > 3:
                self._extract_facts_from_sentence(sentence.text)
        
        nx.write_gpickle(self.graph, self.graph_path)
        print("KG_Tool: Knowledge Graph saved.")

    def query(self, entity: str) -> Optional[str]:
        """Public method to query the graph for facts about an entity."""
        print(f"KG_Tool: Querying Knowledge Graph for entity: '{entity}'...")
        target_node = None
        # Be more forgiving with the search for the node
        search_entity = entity.strip().lower()
        for node in self.graph.nodes():
            if node.strip().lower() == search_entity:
                target_node = node
                break
        
        if not target_node:
            return f"No specific facts found for '{entity}' in my memory."
        
        facts = []
        # Find all nodes connected to the entity node
        for neighbor in self.graph.neighbors(target_node):
            relationship = self.graph[target_node][neighbor].get('label', 'is related to')
            facts.append(f"- {target_node} -> [{relationship}] -> {neighbor}")
        
        if not facts:
            return f"Found entity '{entity}', but no specific relationships are recorded."
        
        return f"Here are the facts I know about '{entity}':\n" + "\n".join(facts)

if __name__ == '__main__':
    print("--- Running Knowledge Graph Tool Demo (SPO Extractor) ---")
    try:
        # On first run, this will be slower as it downloads the spacy model
        kg_tool = KnowledgeGraphTool(base_dir="Database")

        print("\n--- Demo logic running ---")
        
        test_text = "The DSR tool uses Google for searches. The calculator tool, which was built with SymPy, handles trigonometry."
        
        # Update the graph with new text
        kg_tool.update_graph_from_text(test_text)
        
        print("\n--- Testing Knowledge Graph Query ---")
        facts_dsr = kg_tool.query("DSR tool")
        print("\nQuery Result for 'DSR tool':")
        print(facts_dsr)
        
        facts_calc = kg_tool.query("calculator tool")
        print("\nQuery Result for 'calculator tool':")
        print(facts_calc)

        print("\n--- Demo Complete ---")

    except Exception as e:
        import traceback
        print(f"\nDemo failed during setup or execution. Error: {e}")
        traceback.print_exc()