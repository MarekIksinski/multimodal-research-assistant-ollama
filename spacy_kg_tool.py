#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 15:05:00 2025
Updated on Thu Jun 12 18:25:00 2025
@author: marek (w/ Gemini)

This module provides a knowledge graph tool that uses ONLY spaCy.
This version enriches the graph by storing the type of each entity (e.g.,
Person, Organization) as a node attribute.
"""
import spacy
import networkx as nx
import os
from typing import Optional

class SpacyKnowledgeGraphTool:
    """
    Builds and queries a knowledge graph using only spaCy's dependency parsing.
    This version adds entity types (ORG, PERSON, GPE) as attributes to the nodes.
    """

    def __init__(self, base_dir: str = "Database", max_length: int = 2000000):
        print("--- Initializing spaCy-Only Knowledge Graph Tool ---")
        os.makedirs(base_dir, exist_ok=True)
        self.graph_path = os.path.join(base_dir, "spacy_knowledge_graph.gpickle")
        self.nlp = self._load_spacy_model()
        # Set a higher max_length to handle long documents from web scraping
        self.nlp.max_length = max_length 
        self.graph = self._load_or_create_graph()
        print(f"--- spaCy-Only KG Tool Initialized (max_length set to {self.nlp.max_length:,} chars) ---")

    def _load_spacy_model(self):
        """Loads the spaCy model, downloading if necessary."""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("SKG_Tool: Spacy 'en_core_web_sm' model not found. Downloading...")
            spacy.cli.download("en_core_web_sm")
            return spacy.load("en_core_web_sm")

    def _load_or_create_graph(self) -> nx.Graph:
        """Loads the knowledge graph from file or creates a new one."""
        if os.path.exists(self.graph_path):
            return nx.read_gpickle(self.graph_path)
        else:
            return nx.Graph()

    def _get_full_phrase(self, token):
        """Extracts the full noun phrase from a token, including compounds and modifiers."""
        sub_tokens = [child for child in token.lefts if child.dep_ in ('compound', 'amod', 'det')]
        sub_tokens.append(token)
        sub_tokens.extend([child for child in token.rights if child.dep_ in ('compound')])
        return " ".join(t.text for t in sorted(sub_tokens, key=lambda x: x.i))

    # --- MODIFIED: This method now extracts and stores entity types ---
    def update_graph_from_text(self, text: str):
        """
        Processes text using spaCy to find facts and adds them to the graph,
        including the type of each entity (e.g., PERSON, ORG).
        """
        print(f"SKG_Tool: Updating graph from text using dependency parsing...")
        new_facts_found = []

        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            doc = self.nlp(para)

            # First, find all named entities in the paragraph and map them
            entity_map = {ent.text.strip(): ent.label_ for ent in doc.ents}

            for sent in doc.sents:
                root = sent.root
                if root.pos_ != 'VERB':
                    continue

                predicate = root.lemma_
                subjects = []
                objects = []

                for child in root.children:
                    if child.dep_ in ('nsubj', 'nsubjpass'):
                        subjects.append(self._get_full_phrase(child))
                    elif child.dep_ in ('dobj', 'attr'):
                        objects.append(self._get_full_phrase(child))
                    elif child.dep_ == 'prep':
                        preposition = child.text
                        for pobj in [p for p in child.children if p.dep_ == 'pobj']:
                            full_predicate = f"{predicate} {preposition}"
                            for subject_phrase in subjects:
                                obj_phrase = self._get_full_phrase(pobj)
                                
                                # Add nodes with their types before adding the edge
                                self.graph.add_node(subject_phrase, type=entity_map.get(subject_phrase, 'CONCEPT'))
                                self.graph.add_node(obj_phrase, type=entity_map.get(obj_phrase, 'CONCEPT'))
                                self.graph.add_edge(subject_phrase, obj_phrase, label=full_predicate)
                                new_facts_found.append(f"({subject_phrase}) -> [{full_predicate}] -> ({obj_phrase})")
                
                if subjects and objects:
                    for subject_phrase in subjects:
                        for obj_phrase in objects:
                            self.graph.add_node(subject_phrase, type=entity_map.get(subject_phrase, 'CONCEPT'))
                            self.graph.add_node(obj_phrase, type=entity_map.get(obj_phrase, 'CONCEPT'))
                            self.graph.add_edge(subject_phrase, obj_phrase, label=predicate)
                            new_facts_found.append(f"({subject_phrase}) -> [{predicate}] -> ({obj_phrase})")
        
        num_added = len(new_facts_found)
        if num_added > 0:
            print(f"    + Added {num_added} new facts to the knowledge graph.")
            sample_size = 4
            if num_added > sample_size:
                print(f"      (Showing a sample of {sample_size}...)")
            for fact in new_facts_found[:sample_size]:
                print(f"        - {fact}")
            if num_added > sample_size:
                print("        - ...")
        else:
            print("    + No new facts were added from this text.")

        nx.write_gpickle(self.graph, self.graph_path)
        print("SKG_Tool: Knowledge Graph saved.")

    # --- MODIFIED: This method now displays the entity type in the output ---
    def query(self, entity: str) -> Optional[str]:
        """Queries the graph for facts about a specific entity and shows its type."""
        print(f"SKG_Tool: Querying graph for entity: '{entity}'...")
        target_node = None
        search_entity = entity.strip().lower()
        for node in self.graph.nodes():
            if search_entity in node.strip().lower():
                target_node = node
                break
        
        if not target_node:
            return f"No specific facts found for '{entity}'."
        
        facts = []
        # Get the type of the target node itself
        target_node_type = self.graph.nodes[target_node].get('type', 'CONCEPT')
        facts.append(f"Entity: {target_node} (Type: {target_node_type})")
        
        for neighbor in self.graph.neighbors(target_node):
            relationship = self.graph[target_node][neighbor].get('label', 'is related to')
            neighbor_type = self.graph.nodes[neighbor].get('type', 'CONCEPT')
            facts.append(f"- [{relationship}] -> {neighbor} (Type: {neighbor_type})")
        
        if len(facts) <= 1:
            return f"Found entity '{target_node}' (Type: {target_node_type}), but no relationships are recorded."
        
        return "Facts about " + "\n".join(facts)

    def extract_keyphrases(self, text: str) -> list[str]:
        """
        Extracts keyphrases using spaCy's NER and noun chunking.
        """
        doc = self.nlp(text)
        phrases = set()
        
        # Add named entities (e.g., "Google", "Ukraine")
        for ent in doc.ents:
            phrases.add(ent.text.strip())
        
        # Add noun chunks (e.g., "the Ukrainian battlefield", "a new AI model")
        for chunk in doc.noun_chunks:
            # Add some filtering to avoid very short or generic chunks like pronouns
            if len(chunk.text.split()) <= 4 and chunk.root.pos_ != 'PRON':
                phrases.add(chunk.text.strip())

        # Sort the results to prioritize longer, more specific phrases
        sorted_phrases = sorted(list(phrases), key=len, reverse=True)
        return sorted_phrases

if __name__ == '__main__':
    print("--- Running spaCy-Only KG Tool Demo ---")
    try:
        skg_tool = SpacyKnowledgeGraphTool(base_dir="Database")
        # Clear the graph for a clean demo run
        skg_tool.graph.clear()
        
        test_text = "The company Google, based in California, announced a new AI model called 'Gemini'. The announcement was made by CEO Sundar Pichai."
        skg_tool.update_graph_from_text(test_text)
        
        print("\n--- Testing Knowledge Graph Query ---")
        facts_google = skg_tool.query("Google")
        print("\nQuery Result for 'Google':")
        print(facts_google)

        facts_sundar = skg_tool.query("Sundar Pichai")
        print("\nQuery Result for 'Sundar Pichai':")
        print(facts_sundar)

    except Exception as e:
        import traceback
        print(f"\nDemo failed during setup or execution. Error: {e}")
        traceback.print_exc()