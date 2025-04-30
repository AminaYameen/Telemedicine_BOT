# Telemedicine_BOT Backend 🩺

## Overview
The Telemedicine_BOT Backend powers an AI-driven telemedicine chatbot designed to assist users with hospital-related queries. Built using FastAPI, LangChain, and Chainlit, the bot combines large language models (LLMs), real-time search, and vector-based memory to deliver a smart, interactive healthcare assistant.

## Features
- 🧠 AI Chatbot powered by LangChain and LangGraph
- 🏥 Hospital search using text data and embeddings
- 🔍 External search via Tavily API
- 🗂️ Semantic memory with FAISS
- 🖥️ Interactive chatbot UI via Chainlit
- ⚙️ Backend powered by FastAPI + Uvicorn

## Requirements
- Python 3.13+
- uv – For fast dependency management
- FastAPI
- Uvicorn
- LangChain
- LangGraph
- Chainlit
- FAISS (faiss-cpu)
- Google Generative AI
- Tavily API
- python-dotenv
- pydantic
- beautifulsoup4

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone https://github.com/masfana016/Telemedicine_BOT.git
   cd Telemedicine_BOT/Backend

1. **Install uv in powershell**
   ```bash
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
2. **Run in Project terminal**
   ```bash
   uv init
3. **Install Dependencies with uv**
   ```bash
   uv add requirement_name
5. **Create a .env File In the Backend directory, create a .env file with the following:**
   ```bash
   GOOGLE_API_KEY=your_google_api_key
   TAVILY_API_KEY=your_tavily_api_key
