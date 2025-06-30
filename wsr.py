#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the 'process_query' function for the wsr (quick search) tool.
It now processes the top 5 web search results for more comprehensive context,
and returns a structured dictionary with a credibility assessment.
"""
import json
from web_utils import WebUtils # Assumes web_utils.py is in the same directory

# --- Configuration for this specific tool ---
WSR_CONFIG = {
    "GOOGLE_API_KEY": "XYZ", # Replace with your key
    "GOOGLE_CSE_ID": "XYZ", # Replace with your CSE ID
    "OLLAMA_API_URL": "http://localhost:11434/api/generate",
    "MODELS": {
        "summarizer": { # The only model needed for this simple tool
            "name": "gemma3:4b-it-qat",
            "options": {"temperature": 0.2, "num_ctx": 64192}
        },
        # --- NEW: Added a simple language detector model config ---
        "detector": {
            "name": "gemma3:12b-it-qat",
            "options": {"temperature": 0.1, "num_ctx": 8000}
        },
        "relevance_checker": { # NEW
            "name": "gemma3:12b-it-qat",
            "options": {"temperature": 0.1, "num_ctx": 8000}
        }
    }
}

# --- Main Tool Function ---
def process_query(query: str) -> dict:
    """
    The main function for the wsr tool. It now searches the top 5 results,
    scrapes them, summarizes the combined content, and returns a structured dictionary.
    """
    print(f"--- [WSR] Processing query: '{query}' ---")
    
    web_utils = WebUtils(WSR_CONFIG)

    # Step 1: Detect language to perform a better search
    print("[WSR] Detecting query language...")
    lang_prompt = f"What is the two-letter ISO 639-1 language code of the following query? Respond with ONLY the two-letter code (e.g., 'en', 'zh', 'uk').\n\nQuery: '{query}'"
    lang_code = web_utils.query_ollama(lang_prompt, model_key="detector").lower().strip()
    if len(lang_code) != 2 or not lang_code.isalpha():
        print(f"[WSR] Warning: Could not determine language. Defaulting to English.")
        lang_code = 'en'

    # --- MODIFIED: Fetch top 5 results ---
    urls = web_utils.google_search(query, num_results=5, language=lang_code)
    
    if not urls:
        return {
            "summary": "Could not find any relevant information from the web search.",
            "source_type": "Standard Web Search",
            "credibility": "Low"
        }

    # --- MODIFIED: Loop through all URLs and collect content ---
    print(f"[WSR] Found {len(urls)} URLs to process.")
    all_relevant_content = []
    processed_urls = []
    for url in urls:
        content = web_utils.scrape_webpage(url)
        if content:
            if web_utils.is_relevant(content, query):
                all_relevant_content.append(content)
            processed_urls.append(url)
    
    if not all_relevant_content:
        return {
            "summary": f"Found {len(urls)} sources but could not scrape content from any of them.",
            "source_type": "Standard Web Search",
            "credibility": "Low"
        }

    # Combine all scraped text into a single block
    combined_content = "\n\n--- Next Source ---\n\n".join(all_relevant_content)

    print(f"[WSR] Summarizing combined content from {len(all_relevant_content)} sources...")
    summary_prompt = f"Based on the user's query '{query}', provide a concise, direct summary of the key information found in the following texts from multiple sources:\n\nTEXTS:\n{combined_content}"
    summary = web_utils.query_ollama(summary_prompt, model_key="summarizer")

    # Return the final structured dictionary
    return {
        "summary": summary if summary else "Could not generate a summary from the collected content.",
        "source_type": f"Standard Web Search ({len(all_relevant_content)} sources)",
        "credibility": "Medium"
    }

if __name__ == '__main__':
    print("--- Running WSR (Standard Search) Tool Demo ---")
    test_query = "What are the main differences between Python and Javascript?"
    result = process_query(test_query)
    print("\n--- Result for Test Query ---")
    print(json.dumps(result, indent=2))
