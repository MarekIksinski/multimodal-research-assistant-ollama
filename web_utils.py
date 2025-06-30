#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 14:00:00 2025
@author: marek (w/ Gemini)

This is a central utility module to handle all interactions with external
services like Google Search and the Ollama LLM API. It also contains
the web scraper. This helps to avoid code duplication.
"""
import requests
import json
import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig
from typing import Dict, Any

class WebUtils:
    """A utility class for web searches, scraping, and LLM calls."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the WebUtils class with necessary configurations.
        
        Args:
            config (Dict): A dictionary containing API keys and model configs.
        """
        self.google_api_key = config.get("GOOGLE_API_KEY")
        self.google_cse_id = config.get("GOOGLE_CSE_ID")
        self.ollama_api_url = config.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
        self.models_config = config.get("MODELS")

    def google_search(self, query: str, num_results: int = 4, language: str = 'en') -> list:
        """Performs a Google Custom Search."""
        print(f"[WebUtils] Google Search: '{query}' (Lang: {language})")
        if not self.google_api_key or not self.google_cse_id:
            print("[WebUtils] ERROR: Google API Key or CSE ID is not configured.")
            return []
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "q": query,
                "key": self.google_api_key,
                "cx": self.google_cse_id,
                "num": num_results,
                "lr": f"lang_{language}",
                "hl": language
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            results = response.json()
            return [item['link'] for item in results.get('items', [])]
        except requests.RequestException as e:
            print(f"[WebUtils] Error during Google search: {e}")
            return []

    async def _async_scrape_webpage(self, url: str) -> str:
        """Scrapes a single webpage asynchronously."""
        print(f"[WebUtils] Scraping URL: {url}")
        run_config = CrawlerRunConfig(word_count_threshold=100, remove_overlay_elements=True)
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url, config=run_config)
                if result and result.markdown:
                    content = result.markdown.strip()
                    # --- NEW: Add a print statement for the character count ---
                    print(f"[WebUtils] Scraped {len(content):,} characters from page.")
                    return content
                else:
                    return ""

        except Exception as e:
            print(f"[WebUtils] Error scraping {url}: {e}")
            return ""
            
    def scrape_webpage(self, url:str) -> str:
        """Synchronous wrapper for the async scraper."""
        return asyncio.run(self._async_scrape_webpage(url))

    def query_ollama(self, prompt: str, model_key: str) -> str:
        """Queries the Ollama LLM using a key to get model and options from config."""
        model_config = self.models_config.get(model_key)
        if not model_config:
            print(f"[WebUtils] ERROR: Model key '{model_key}' not found in configuration.")
            return ""

        payload = {
            "model": model_config["name"],
            "prompt": prompt,
            "stream": False,
            "options": model_config["options"]
        }
        try:
            response = requests.post(self.ollama_api_url, json=payload, timeout=None)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.RequestException as e:
            print(f"[WebUtils] Error querying Ollama API: {e}")
            return ""
        
    def is_relevant(self, text_content: str, user_query: str) -> bool:
        """
        Uses a fast LLM call to check if a text snippet is relevant to the user's query.
        """
        print("[WebUtils] Performing relevance check on content...")
        
        # We only use the beginning of the text to keep this check fast
        snippet = text_content[:2000]

        prompt = f"""
        You are a relevance checking assistant. Based on the user's query, is the following text content relevant to answering it?
        Your entire response must be ONLY the word YES or NO.

        **User Query:** '{user_query}'

        **Text Content Snippet:**
        '{snippet}'
        """
        # Use a fast, small model for this simple task. We'll configure it in the tool's config.
        response = self.query_ollama(prompt, model_key="relevance_checker").strip().upper()
        
        if "YES" in response:
            print("[WebUtils] Content deemed RELEVANT.")
            return True
        else:
            print("[WebUtils] Content deemed IRRELEVANT. Skipping.")
            return False