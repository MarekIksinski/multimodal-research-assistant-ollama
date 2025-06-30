#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 23:43:58 2025
Updated on Tue Jun 10 22:00:00 2025

@author: marek (w/ Gemini)

This is the final, fully integrated script for a multi-turn, multimodal chatbot
with a full suite of tools and session management.
"""
import requests
import json
import re
import base64
import os
import sys
from datetime import date
from typing import Optional, List, Dict, Any

# --- All our custom tools and managers ---
from conversation_manager import ConversationManager
from csr import ConversationalSearchAgent
from kgr import KnowledgeGraphTool
from document_search_tool import DocumentSearchTool
from calculator_tool import execute_calculator_tool
from statistics_tool import execute_statistics_tool
from wsr import process_query as execute_web_search
from dsr import execute_deep_search

# ==============================================================================
# --- Configuration & Global Initializations ---
# ==============================================================================

OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
MODEL_NAME = 'gemma3:12b-it-qat'
model_options = {'temperature': 0.3, 'num_ctx': 16384, 'top_p': 0.85}

# --- Initialize all tools and managers once at the start ---
try:
    print("Initializing all agent systems...")
    manager = ConversationManager(base_dir="Database")
    csr_agent = ConversationalSearchAgent(conversation_db_path="Database/conversation_vectors")
    kg_tool = KnowledgeGraphTool(base_dir="Database")
    doc_search_tool = DocumentSearchTool(db_path="Database/pdf_knowledge_base")
    print("All systems initialized successfully.")
except Exception as e:
    print(f"\nFATAL: Could not initialize required agents. Exiting. Error: {e}")
    exit(1)

# --- System Prompt and Utility Functions ---

def get_system_prompt():
    # ... (This function is unchanged from the last version with all tools)
    today_iso = date.today().isoformat()
    return f"""
    Current Date: {today_iso}.    
    You are a helpful and concise AI assistant. Your primary goal is to answer the user's questions accurately.

    Tool Usage:
    You have access to two special tools to assist you:

    1. document_search(query):
       Use to search for information within a private library of pre-loaded PDF documents, such as scientific papers or manuals. Use this when the user asks a question about a specific document or topic that would be contained in such a library (e.g., "What did the paper on watermarking say about STA-1?").

    2. web_search(query):
       Use this for up-to-date information (e.g., news, current events) or specific facts you don't know.

    3. conversation_search(query):
       Use this tool if the user asks a question about what was said earlier, refers to a past topic, or asks you to remember or summarize previous parts of our conversation. This tool is for accessing your long-term memory.

    4. calculator(operation, **kwargs):
       Use this for any precise mathematical calculation. Do not try to calculate complex math yourself.
       The 'operation' argument specifies the type of calculation. Available operations are:
       - "evaluate": For general arithmetic and expressions (e.g., "15 * sin(pi/2)"). Use 'expression' as the argument.
       - "differentiate": For derivatives. Use 'expression' and 'symbol' as arguments.
       - "expand_polynomial": For expanding polynomials. Use 'expression' as the argument.
       - "solve_equation": To solve an equation for a variable. The equation is assumed to be equal to zero. Use 'equation' and 'symbol' as arguments.

    5. statistics(operation, data):
        Use for statistical calculations on a list of numbers. The 'data' argument must be a JSON array of numbers.
        Available operations are: "mean", "median", "mode", "stdev", "variance", "full_summary".

    6. deep_search(query):
       Use for COMPLEX, open-ended questions that require research from multiple sources. Use this for questions like "Explain...", "Summarize...", "Compare and contrast...", or "What are the arguments for...". This tool is slow but very thorough.

    7. knowledge_graph_query(entity):
       Use for specific, factual questions about a key person, place, or concept we have already discussed (e.g., "What do you know about the 'calculator tool'?", "Tell me the facts about 'Warkworth Castle'").



    CRITICAL INSTRUCTION FOR TOOL USE:
    If you decide to use a tool, your entire response for that turn MUST ONLY BE A JSON OBJECT in the following format.
    DO NOT include any other text before or after the JSON object.

    {{
      "tool_call": {{
        "name": "<tool_name_here>",
        "arguments": {{
          "query": "<the specific and effective search query string you want to use>"
        }}
      }}
    }}

    Example for Calculator:
    {{
      "tool_call": {{
        "name": "calculator",
        "arguments": {{
          "operation": "differentiate",
          "expression": "x**2 + 2*x",
          "symbol": "x"
        }}
      }}
    }}

    Example for Statistics:
    {{
      "tool_call": {{
        "name": "statistics",
        "arguments": {{
          "operation": "mean",
          "data": [10, 15, 20, 25]
        }}
      }}
    }}
    
    If you can answer directly without needing a tool, provide a direct answer.
    
    """

def image_to_base64(image_path: str) -> Optional[str]:
    # ... (Unchanged)
    if not os.path.exists(image_path): return None
    try:
        with open(image_path, "rb") as f: return base64.b64encode(f.read()).decode('utf-8')
    except Exception: return None

def format_duration_ns(nanoseconds: int) -> str:
    # ... (Unchanged)
    if nanoseconds is None: return "N/A"
    if nanoseconds < 1e3: return f"{int(nanoseconds)} ns"
    if nanoseconds < 1e6: return f"{nanoseconds / 1e3:.3f} µs"
    if nanoseconds < 1e9: return f"{nanoseconds / 1e6:.3f} ms"
    return f"{nanoseconds / 1e9:.3f} s"

def synthesize_final_answer(messages: list, evidence_package: str, original_query: str, original_image_prompt: str, has_image: bool, tool_image_data: Optional[str] = None) -> tuple:
    """
    Calls the LLM with a sophisticated prompt to synthesize a final answer
    based on structured, credibility-tagged evidence.
    """
    print("Assistant: Synthesizing final answer from tool results...")
    
    synthesis_prompt_intro = f"You are an expert research analyst. Your task is to synthesize a comprehensive answer to the user's original query using the provided information from your tools.\n\n**Provided Information:**\n{evidence_package}\n\n"
    
    if has_image:
        synthesis_prompt_conclusion = f"**Instructions for Synthesis:**\n1. Give the most weight to information from 'High' credibility sources.\n2. Use 'Medium' credibility sources for context.\n3. Use 'Low' credibility sources with caution.\n4. If sources conflict, prioritize the source with higher credibility.\n5. Based on this, synthesize a single, well-structured answer to the user's original question about the provided image: \"{original_image_prompt}\""
    else:
        synthesis_prompt_conclusion = f"**Instructions for Synthesis:**\n1. Give the most weight to information from 'High' credibility sources.\n2. Use 'Medium' credibility sources for context.\n3. Use 'Low' credibility sources with caution.\n4. If sources conflict, prioritize the source with higher credibility.\n5. Based on this, synthesize a single, well-structured answer to the user's original query: \"{original_query}\""

    synthesis_message = {'role': 'user', 'content': synthesis_prompt_intro + synthesis_prompt_conclusion}
    
    if tool_image_data:
        synthesis_message['images'] = [tool_image_data]

    # Provide the main conversational history for context
    contextual_history = messages
    
    final_payload = {
        "model": MODEL_NAME,
        "messages": contextual_history + [synthesis_message],
        "options": model_options,
        "stream": False
    }
    
    # This should now use the central WebUtils for consistency, but a direct call is fine for now
    final_response_obj = requests.post(OLLAMA_ENDPOINT, json=final_payload)
    final_response_obj.raise_for_status()
    final_response_data = final_response_obj.json()
    
    final_answer = final_response_data.get('message', {}).get('content', 'Sorry, I encountered an error after processing the tool results.')
    return final_answer, final_response_data

def generate_and_save_summary(conv_id, history):
    # ... (Unchanged, but now passed the manager instance)
    print("\nAssistant: Generating session summary...")
    transcript = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history if msg['role'] in ['user', 'assistant']])
    summary_prompt = f"You are a summarization assistant... TRANSCRIPT:\n{transcript}"
    summary_payload = {"model": MODEL_NAME, "messages": [{'role': 'user', 'content': summary_prompt}], "options": {'temperature': 0.0}, "stream": False}
    try:
        response = requests.post(OLLAMA_ENDPOINT, json=summary_payload)
        response.raise_for_status()
        summary_text = response.json().get('message', {}).get('content', '')
        if summary_text:
            manager.update_conversation_summary(conv_id, summary_text)
    except Exception as e:
        print(f"An error occurred during summary generation: {e}")

# ==============================================================================
# --- NEW: Application Flow Functions ---
# ==============================================================================

def chat_session(conversation_id: int, messages: list, conv_name: str):
    """Contains the main loop for a single chat conversation."""
    print(f"\nChatbot ready. Conversation '{conv_name}' (ID: {conversation_id}).")
    print("\n--- Commands ---\nImage Input:  image: /path/to/image.jpg [prompt]\nEnd Chat:     exit:\nQuit Program: quit:\nSet Param:    set: [param] [value]\n" + "-"*30)

    while True:
        try:
            user_input_full = input("You: ")
            
            # --- Command Handling ---
            user_command = user_input_full.lower().strip()
            if user_command == 'quit:':
                generate_and_save_summary(conversation_id, messages)
                print("Exiting chatbot. Goodbye!")
                sys.exit()
            elif user_command == 'exit:':
                generate_and_save_summary(conversation_id, messages)
                print("Ending current chat. Returning to main menu...")
                return # Go back to the main menu
            elif user_command.startswith('set:'):
                try:
                    parts = user_input_full.strip()[4:].strip().split()
                    if len(parts) == 2:
                        param_name, param_value_str = parts
                        allowed_params = ["temperature", "top_p", "num_ctx"]
                        if param_name in allowed_params:
                            model_options[param_name] = float(param_value_str) if param_name != "num_ctx" else int(param_value_str)
                            print(f"Parameter '{param_name}' updated to {model_options[param_name]} for this session.")
                        else:
                            print(f"Invalid parameter. Allowed: {allowed_params}")
                    else:
                        print("Invalid format. Use: set: [parameter] [value]")
                except (ValueError, IndexError):
                    print("Invalid value for parameter.")
                continue # Wait for next user input

            # --- Input Parsing (Text and Image) ---
            user_text_prompt, base64_image_data = (user_input_full, None)
            if user_command.startswith("image:"):
                try:
                    _c, path_and_prompt = user_input_full.split(":", 1)
                    path_match = re.match(r'^(.*?(\.jpg|\.jpeg|\.png|\.webp))\s*', path_and_prompt.strip(), re.IGNORECASE)
                    if path_match:
                        image_path = path_match.group(1).strip()
                        prompt_text = path_and_prompt[len(path_match.group(0)):].strip()
                        if os.path.exists(image_path):
                            base64_image_data = image_to_base64(image_path)
                            user_text_prompt = prompt_text if prompt_text else "Describe this image."
                except Exception as e:
                    print(f"Could not parse image command: {e}")

            current_user_message = {'role': 'user', 'content': user_text_prompt}
            if base64_image_data: current_user_message['images'] = [base64_image_data]
            messages.append(current_user_message)

            # --- Primary LLM Call (Decision Maker) ---
            print("\nAssistant: Thinking...")
            payload = {"model": MODEL_NAME, "messages": messages, "options": model_options, "stream": False}
            response = requests.post(OLLAMA_ENDPOINT, json=payload)
            response.raise_for_status()
            response_data = response.json()
            assistant_response_content = response_data.get('message', {}).get('content', '').strip()

            final_answer = assistant_response_content
            tool_call_detected = False

            # --- Tool Call Parsing ---
            json_string = None
            if '```json' in assistant_response_content:
                match = re.search(r'```json\s*(\{.*?\})\s*```', assistant_response_content, re.DOTALL)
                if match: json_string = match.group(1)
            elif assistant_response_content.strip().startswith('{'):
                json_string = assistant_response_content.strip()

            if json_string:
                try:
                    tool_call_data = json.loads(json_string)
                    if isinstance(tool_call_data, dict) and "tool_call" in tool_call_data:
                        tool_call_detected = True
                        # --- Tool Execution ---
                        tool_info = tool_call_data["tool_call"]
                        tool_name = tool_info.get("name")
                        tool_args = tool_info.get("arguments", {})
                        messages.append({'role': 'assistant', 'content': assistant_response_content})
                        
                        evidence_package = "Error: Tool execution failed."
                        tool_image_data = None
                        
                        # --- Tool Routing Logic ---
                        if tool_name in ["web_search", "deep_search"]:
                            tool_query = tool_args.get('query')
                            print(f"Assistant: Decided to use {tool_name}. Query: \"{tool_query}\"")
                            tool_result = execute_deep_search(tool_query) if tool_name == "deep_search" else execute_web_search(tool_query)
                            summary = tool_result.get("summary", "No summary.")
                            source_type = tool_result.get("source_type", "Unknown")
                            credibility = tool_result.get("credibility", "Unknown")
                            evidence_package = f"* Source Type: {source_type} (Credibility: {credibility})\n  Summary: {summary}"
                        
                        elif tool_name == "document_search":
                            tool_query = tool_args.get('query')
                            print(f"Assistant: Decided to use document_search. Query: \"{tool_query}\"")
                            result = doc_search_tool.search(tool_query)
                            evidence_package = result.get('text_context', 'No text found.')
                            tool_image_data = result.get('image_base64')
    
                        elif tool_name in ["calculator", "statistics", "knowledge_graph_query", "conversation_search"]:
                            query_arg = tool_args.get("entity") or tool_args.get("query")
                            print(f"Assistant: Decided to use {tool_name} with args: {tool_args}")
                            if tool_name == "calculator": result = execute_calculator_tool(**tool_args)
                            elif tool_name == "statistics": result = execute_statistics_tool(**tool_args)
                            elif tool_name == "knowledge_graph_query": result = kg_tool.query(query_arg)
                            else: result = csr_agent.search(query=query_arg)
                            evidence_package = f"* Tool: {tool_name}\n  Result: {result}"

                        # --- Synthesis after Tool Call ---
                        # THIS IS THE CORRECTED CALL
                        final_answer, response_data = synthesize_final_answer(
                            messages=messages,
                            evidence_package=evidence_package,
                            original_query=user_input_full,
                            original_image_prompt=user_text_prompt,
                            has_image=bool(base64_image_data),
                            tool_image_data=tool_image_data
                        )
                        
                        # --- Selective KG Update ---
                        if tool_name in ["web_search", "deep_search", "document_search"]:
                            print(f"KG_Tool: Updating Knowledge Graph from {tool_name} results...")
                            kg_tool.update_graph_from_text(final_answer)

                except json.JSONDecodeError:
                    print(f"Info: Response looked like JSON but failed to parse. Treating as text.")
                    tool_call_detected = False

            # --- Final output and saving for ALL turns ---
            print(f"\nAssistant:\n{final_answer}")
            messages.append({'role': 'assistant', 'content': final_answer})
            
            print("\nAssistant: Saving conversation to memory...")
            manager.add_turn_to_history(conversation_id, user_text_prompt, final_answer)
            
            # Note: We only update the KG after tool calls now, not here.
            
            print("\n--- Turn Statistics ---")
            print(f"Total Duration: {format_duration_ns(response_data.get('total_duration'))}")
            print("-" * 30)

        except Exception as e:
            print(f"\nAn unexpected error occurred in the chat loop: {e}")
            import traceback
            traceback.print_exc()
            if messages and messages[-1]['role'] == 'user': messages.pop()
            print("-" * 30)

def select_or_create_session():
    """Handles the main menu for starting or loading a conversation."""
    while True:
        print("\n--- Conversation Menu ---")
        print("[1] Start a new conversation")
        print("[2] Load an existing conversation")
        print("[3] Quit Program")
        choice = input("Your choice: ").strip()

        if choice == '1':
            conv_name = input("Enter an optional name for this conversation (or press Enter): ")
            active_name = conv_name if conv_name else f"Chat from {date.today().isoformat()}"
            conv_id = manager.start_new_conversation(name=active_name)
            initial_messages = [{'role': 'system', 'content': get_system_prompt()}]
            return conv_id, initial_messages, active_name
        
        elif choice == '2':
            convos = manager.list_conversations()
            if not convos:
                print("No existing conversations found.")
                continue
            print("\nAvailable conversations:")
            for c in convos:
                print(f"  ID: {c['id']} | Name: {c['name']} | Started: {c['start_time']} | Has Summary: {c['summary_exists']}")
            try:
                load_id_str = input("Enter the ID of the conversation to load: ")
                if not load_id_str.isdigit():
                    print("Invalid ID. Please enter a number.")
                    continue
                
                load_id = int(load_id_str)
                if not any(c['id'] == load_id for c in convos):
                    print("Invalid ID.")
                    continue
                    
                summary = manager.get_conversation_summary(load_id)
                initial_messages = [{'role': 'system', 'content': get_system_prompt()}]
                
                if summary:
                    # --- THIS IS THE CORRECTED LINE ---
                    initial_messages.append({'role': 'assistant', 'content': f"Okay, I've loaded the summary of our previous session to refresh my memory. The key points were: {summary}"})
                    print("Continuity summary has been loaded into my context.")
                
                loaded_conv_name = next((c['name'] for c in convos if c['id'] == load_id), f"Chat from {date.today().isoformat()}")
                return load_id, initial_messages, loaded_conv_name
            except (ValueError, TypeError):
                print("Invalid ID.")
            
        elif choice == '3':
            return None, None, None # Signal to quit
        
        else:
            print("Invalid choice.")

def main():
    """The main application entry point."""
    while True:
        conversation_id, messages, conv_name = select_or_create_session()
        if conversation_id is None:
            print("Goodbye!")
            break
        chat_session(conversation_id, messages, conv_name)

if __name__ == "__main__":
    main()
