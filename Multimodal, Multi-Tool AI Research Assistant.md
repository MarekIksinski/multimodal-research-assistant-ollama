# **Multimodal, Multi-Tool AI Research Assistant**

## **Overview**

This project is a sophisticated, locally-run AI assistant built on Python and the Ollama framework. It goes far beyond a simple chatbot by incorporating a powerful, tool-based architecture that gives it advanced capabilities, including a hybrid memory system, multiple forms of web research, multimodal document analysis, and more. The agent can reason about which tool to use for a given task, learn from its research, and maintain context across multiple sessions.

### **Key Features**

* **Multi-Tool Framework:** The agent can seamlessly decide when to use a variety of specialized tools to answer complex questions.  
* **Session Management:** Supports multiple, persistent conversations with session summaries to maintain context between restarts.  
* **Hybrid Memory System:**  
  * **Conversational Memory:** Remembers past turns using both semantic vector search (for "what did we talk about?") and a structured Knowledge Graph (for "what are the facts about X?").  
  * **Knowledge Acquisition:** Automatically learns and adds new facts to its Knowledge Graph from its own research results and ingested documents.  
* **Advanced Web Research & Analysis:**  
  * **Quick Search:** For fast, single-source fact-checking.  
  * **Deep Search:** An autonomous agent that creates a multi-step research plan, scrapes multiple sources, and synthesizes a comprehensive answer.  
  * **Relevance & Credibility Analysis:** Intelligently checks if web sources are relevant *before* processing and assesses their credibility to weigh evidence in its final answer.  
  * **Multilingual:** Can generate search queries and process results in multiple languages.  
* **Multimodal Document RAG:**  
  * An offline ingestion pipeline processes PDFs, creating both a searchable vector database and a Knowledge Graph from abstracts.  
  * The agent can search this private library, retrieving not only relevant text but also an **image of the page** it came from to answer questions about figures, tables, and graphs.  
* **Analytical Tools:**  
  * **Calculator:** A SymPy-based tool for precise symbolic math, including derivatives and polynomials.  
  * **Statistics:** A tool for performing statistical analysis on datasets.  
* **Dynamic Control:** Users can change the LLM's parameters (like temperature) in real-time during a conversation.

### **Project Structure**

.  
├── Database/                   \# Holds all persistent data  
│   ├── conversations.sqlite    \# Raw text of all chats  
│   ├── conversation\_vectors/   \# ChromaDB for semantic chat search  
│   ├── pdf\_knowledge\_base/     \# ChromaDB for PDF text chunks  
│   ├── knowledge\_graph.gpickle \# The main knowledge graph file  
│   └── pdf\_images/             \# Stores images of ingested PDF pages  
├── pdf/                        \# (Optional) A place to store your PDFs  
│   └── example\_paper.pdf  
├── main.py                     \# The main application entry point  
├── ingest\_papers.py            \# Offline script to process PDFs  
├── web\_utils.py                \# Shared utilities for scraping and API calls  
├── conversation\_manager.py     \# Manages SQLite and chat history  
├── knowledge\_graph\_tool.py     \# Manages the KG (fact extraction, queries)  
├── document\_search\_tool.py     \# Tool for searching the PDF database  
├── csr.py                      \# Tool for semantic search of conversations  
├── dsr.py                      \# Tool for deep, multi-step web research  
├── wsr.py                      \# Tool for quick web searches  
├── calculator\_tool.py          \# The mathematics tool  
├── statistics\_tool.py          \# The statistics tool  
└── pdf-files.txt               \# List of PDFs to be ingested

## **Quick User Guide**

### **1\. Setup & Installation**

**Prerequisites:**

* Python 3.9+  
* An Ollama instance running with a model (e.g., gemma3:12b-it-qat).  
* **Poppler:** A system dependency required for PDF processing.  
  * On Debian/Ubuntu (like your Linux Mint): sudo apt-get install poppler-utils  
  * On macOS (using Homebrew): brew install poppler  
  * On Windows: Requires downloading and adding the Poppler bin directory to your system's PATH.

**Installation:**

1. Clone or download the project files into a single directory.  
2. It is highly recommended to create a Python virtual environment.  
3. Install all required Python libraries:  
   pip install requests spacy networkx pdf2image tinydb langchain-community langchain-huggingface sentence-transformers chromadb crawl4ai sympy

4. Download the necessary spacy model:  
   python \-m spacy download en\_core\_web\_sm

**Configuration:**

1. **API Keys:** Open web\_utils.py and replace the placeholder GOOGLE\_API\_KEY and GOOGLE\_CSE\_ID with your own.  
2. **PDFs:** Create a file named pdf-files.txt in the root directory. Add the full path to each PDF you want the agent to learn from, with one path per line.

### **2\. How to Use**

The system runs in two stages: offline ingestion and the online chat application.

Stage 1: Ingest Your Documents (Offline)  
Before you can chat about your documents, you must run the ingestion pipeline to build the agent's knowledge base.  
python ingest\_papers.py

This will process all the files listed in pdf-files.txt, creating the necessary databases and image files inside the Database directory. You only need to run this when you want to add new documents.

Stage 2: Run the Chatbot  
Once ingestion is complete, start the main application.  
python main.py

You will be greeted with a menu to start a new conversation or load a previous one.

**Available Commands (type in the chat):**

* image: /path/to/image.jpg \[your prompt\] \- Send a local image to the agent for analysis.  
* set: \[parameter\] \[value\] \- Change a model parameter for the current session. (e.g., set: temperature 0.8).  
* exit: \- Ends the current conversation, saves a summary, and returns you to the main menu.  
* quit: \- Saves a summary of the final conversation and closes the entire application.

## **Architectural Deep Dive**

This system is designed as a modular, tool-based agent. The main.py script acts as the central orchestrator or "brain," while specialized modules handle specific tasks.

### **1\. The Core Engine (main.py)**

The main script manages the user-facing chat loop and the agent's decision-making process. On each turn, it sends the conversational history to the LLM. The LLM's primary job is to decide if it can answer directly or if it needs to use a tool. If it chooses a tool, it outputs a structured JSON object, which the main script parses and uses to call the appropriate Python module. This script also contains the session management logic, allowing users to switch between different conversations.

### **2\. The Hybrid Memory System**

The agent has two distinct forms of long-term memory, managed primarily by the ConversationManager and KnowledgeGraphTool.

* **Semantic Memory (Vector Search):** Every conversation turn is vectorized by ConversationManager and stored in a ChromaDB database. The **csr.py** tool allows the agent to perform a semantic search on this history to recall the general context of past conversations, answering questions like "What were we discussing earlier?".  
* **Structured Memory (Knowledge Graph):** The agent builds a Knowledge Graph using networkx, managed by the **knowledge\_graph\_tool.py**. Facts are automatically extracted from research results (wsr/dsr) and PDF abstracts (ingest\_papers.py). This tool allows the agent to query the graph for precise, factual information (e.g., "What is the relationship between X and Y?").

### **3\. Information Retrieval Tools**

The agent has three tiers of tools for gathering new information, all supported by a central web\_utils.py module for common functions.

* **Document Search (document\_search\_tool.py):** This is the agent's private library. It performs an **agentic search** on the vector database created by the ingestion script. It retrieves multiple relevant text chunks, uses an internal LLM to synthesize a high-quality answer, and identifies the most relevant page image to return, making it fully **multimodal**.  
* **Quick Web Search (wsr.py):** This is for fast fact-checking. It scrapes the top 5 Google search results, uses a "relevance checker" to filter out irrelevant pages, and provides a quick summary.  
* **Deep Web Search (dsr.py):** This is a full research agent. It uses an LLM to create a multi-step research plan, executes multiple targeted searches, assesses the credibility of each source, checks for relevance, and finally synthesizes all the weighted evidence into a comprehensive answer.

### **4\. Analytical Tools (calculator\_tool.py, statistics\_tool.py)**

To ensure factual accuracy in quantitative tasks, the agent offloads calculations to dedicated tools. The calculator\_tool.py uses the powerful SymPy library for precise symbolic math, while the statistics\_tool.py uses Python's built-in libraries for data analysis. The LLM's job is simply to formulate the correct expression or data set to pass to the tool.