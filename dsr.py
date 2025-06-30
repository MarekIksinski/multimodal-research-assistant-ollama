#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the 'execute_deep_search' function.
It now uses the central WebUtils module for all web and LLM interactions.
"""
import time
import json
import re
from datetime import date
from web_utils import WebUtils

# --- Configuration for this specific tool ---
DSR_CONFIG = {
    "GOOGLE_API_KEY": "XYZ", # Replace with your key
    "GOOGLE_CSE_ID": "XYZ", # Replace with your CSE ID
    "OLLAMA_API_URL": "http://localhost:11434/api/generate",
    "MODELS": {
        "planner": {"name": "gemma3:12b-it-qat", "options": {"temperature": 0.1, "num_ctx": 4096}},
        "summarizer": {"name": "gemma3:12b-it-qat", "options": {"temperature": 0.3, "num_ctx": 8192}},
        "final_synthesizer": {"name": "gemma3:12b-it-qat", "options": {"temperature": 0.4, "num_ctx": 16384}},
        "relevance_checker": { # NEW
            "name": "gemma3:12b-it-qat",
            "options": {"temperature": 0.0, "num_ctx": 4096}
        }
    }
}

# --- Internal DSR Logic ---
def generate_search_plan(user_query: str, web_utils: WebUtils):
    today_iso = date.today().isoformat()
    """Uses an LLM via WebUtils to create a research plan."""
    print("[DSR] Generating research plan...")
    prompt = f"""
    Current Date: {today_iso}.    
    You are an expert research planner. Analyze the user's question and create a structured research plan.
    Decompose the question into 2-3 essential sub-questions.
    For EACH component, generate 1 specific Google search query and its two-letter ISO 639-1 language code.
    Return ONLY a valid JSON object in the format:
    {{"research_plan": [{{"component": "description", "search_query": "query", "language": "lang_code"}}]}}
    User Question: {user_query}
    """
    response = web_utils.query_ollama(prompt, model_key="planner")
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match: raise json.JSONDecodeError("No JSON object found", response, 0)
        plan_data = json.loads(json_match.group(0))
        if "research_plan" in plan_data:
            plan_steps = [(item['search_query'], item['language']) for item in plan_data['research_plan']]
            print(f"[DSR] Research plan generated with {len(plan_steps)} steps.")
            return plan_steps
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"[DSR] Warning: Could not parse research plan ({e}). Falling back.")
    return [(user_query, 'en')]

def synthesize_final_answer(summaries_with_credibility: list, user_query: str, web_utils: WebUtils):
    """
    Synthesizes all gathered information into a final answer,
    weighing evidence based on source credibility.
    """
    print("[DSR] Synthesizing final answer from weighted sources...")

    # Format the collected information for the final prompt
    provided_info = "\n\n".join(summaries_with_credibility)
    
    prompt = f"""
    You are an expert research analyst. Your task is to synthesize a comprehensive answer to the user's query using the provided information from various sources. Each source has an associated credibility level.

    **User Query:** '{user_query}'

    **Provided Information:**
    {provided_info}

    **Instructions for Synthesis:**
    1. Give the most weight to information from 'High' credibility sources. These should form the foundation of your answer.
    2. Use 'Medium' credibility sources to provide context or supporting details.
    3. Use 'Low' credibility sources with caution. Only mention them if they provide a unique perspective, and explicitly state their nature (e.g., "According to a forum post...").
    4. If sources conflict, prioritize the information from the source with higher credibility. Point out the discrepancy if it is significant.
    5. Synthesize these points into a single, well-structured, and comprehensive answer.
    """
    return web_utils.query_ollama(prompt, model_key="final_synthesizer")

# --- Main Tool Function ---
def execute_deep_search(query: str) -> dict:
    """
    The main entry point for the deep search tool.
    This version has the corrected indentation for the relevance check.
    """
    print(f"\n--- [DSR] Starting Deep Search for: '{query}' ---")
    
    web_utils = WebUtils(DSR_CONFIG)
    search_plan = generate_search_plan(query, web_utils)
    
    all_source_data = []
    processed_urls = set()
    URLS_PER_QUERY = 10

    print(f"[DSR] Executing research plan with {len(search_plan)} steps...")

    for i, (s_query, lang) in enumerate(search_plan):
        print(f"\n[DSR] Executing Plan Step {i+1}/{len(search_plan)}: '{s_query}'")
        urls = web_utils.google_search(s_query, num_results=URLS_PER_QUERY, language=lang)
        if not urls: continue

        for url in urls:
            if url in processed_urls:
                print(f"[DSR] Skipping already processed URL: {url}")
                continue
            
            content = web_utils.scrape_webpage(url)
            if content:
                # --- CORRECTED LOGIC: The entire analysis is now inside this 'if' block ---
                if web_utils.is_relevant(content, query):
                    print(f"[DSR] Analyzing relevant content from: {url}")
                    
                    analysis_prompt = f"""
                    You are a research analyst. The user's main research question is: '{query}'.
                    Based on this question and the text below, perform two tasks:
                    1. Write a concise summary of the key information relevant to the user's query.
                    2. Assess the credibility of the source text as 'High', 'Medium', or 'Low'.

                    Return ONLY a valid JSON object in the format:
                    {{"summary": "your_summary_here", "credibility": "your_rating_here"}}

                    TEXT:
                    {content[:10000]}
                    """
                    
                    response_str = web_utils.query_ollama(analysis_prompt, model_key="summarizer")
                    
                    try:
                        json_string = None
                        json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
                        if json_match:
                            json_string = json_match.group(0)

                        if not json_string:
                             raise AttributeError("No JSON object found in LLM response for analysis.")

                        analysis_result = json.loads(json_string)
                        summary = analysis_result.get("summary", "Summary could not be generated.")
                        credibility = analysis_result.get("credibility", "Medium").capitalize()
                        
                        print(f"[DSR] Source Credibility Assessed as: {credibility}")
                        
                        formatted_source_data = (
                            f"Source: {url} (Credibility: {credibility})\n"
                            f"Summary of Content: {summary}\n"
                        )
                        all_source_data.append(formatted_source_data)
                        
                    except (json.JSONDecodeError, AttributeError, KeyError) as e:
                        print(f"[DSR] Warning: Could not parse analysis JSON ({e}). Using content as summary with Medium credibility.")
                        all_source_data.append(
                            f"Source: {url} (Credibility: Medium)\n"
                            f"Summary of Content: {response_str}\n"
                        )
                # --- End of indented block ---
            
            processed_urls.add(url)
            time.sleep(1)

    if not all_source_data:
        return {"summary": "Found web sources, but none were deemed relevant after skimming.", "source_type": "Deep Search", "credibility": "Low"}

    final_answer_text = synthesize_final_answer(all_source_data, query, web_utils)
    
    print("--- [DSR] Deep Search Complete ---")
    return {
        "summary": final_answer_text,
        "source_type": "Deep Search Synthesis",
        "credibility": "High"
    }
