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
- langchain-google-genai
- python-dotenv
- pydantic
- beautifulsoup4
- bcrypt
- langchain-community
- langchain-core
- sqlmodel
- sqlalchemy

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone https://github.com/masfana016/Telemedicine_BOT.git
   cd Telemedicine_BOT/Backend

1. **Install uv in powershell** https://docs.astral.sh/uv/getting-started/installation/
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
   DB_URI=your_database_URI

## Running the Application
1. **Start the FastAPI Backend**
   ```bash
   uvicorn main:app --port 8001 --reload
2. **Launch the Chainlit UI**
   Chainlit provides a friendly web interface to interact with the AI assistant:
   ```bash
   uv run chainlit run main.py --port 8003
After running the command, open your browser and go to http://localhost:8000 to chat with the bot.

## Project Structure
   ```text
      Backend/
   ├── main.py               # FastAPI + Chainlit logic
   ├── hospitals.txt         # Hospital database (text-based)
   ├── .env                  # Secret API keys
   ├── chainlit.md           # Instructions shown in Chainlit UI
   ├── .chainlit/            # Chainlit UI config (optional)
   ├── pyproject.toml        # Project dependencies
   ├── uv.lock               # Project dependencies
   └── README.md
```

## Tech Stack
- FastAPI: High-performance API framework
- Uvicorn: Fast ASGI server
- LangChain & LangGraph: AI orchestration and control flow
- FAISS: Embedding-based memory for semantic search
- Google Generative AI: LLM for answering user queries
- Tavily API: Real-time web search API
- Chainlit: UI for conversational AI applications
- uv: Lightweight Python package/environment manager

## Contributing
Want to contribute? Feel free to fork the repo, suggest improvements, or open issues on the GitHub repository.

## License
This project is licensed under the MIT License.
