Telemedicine_BOT Backend
🩺 OverviewThe Telemedicine_BOT Backend powers an AI-driven telemedicine chatbot designed to assist users with hospital-related queries. Built using FastAPI, LangChain, and Chainlit, the bot combines large language models (LLMs), real-time search, and vector-based memory to deliver a smart, interactive healthcare assistant.
✅ Features

🧠 AI Chatbot powered by LangChain and LangGraph  
🏥 Hospital search using text data and embeddings  
🔍 External search via Tavily API  
🗂️ Semantic memory with FAISS  
🖥️ Interactive chatbot UI via Chainlit  
⚙️ Backend powered by FastAPI + Uvicorn

📦 Requirements

Python 3.8+  
uv – For fast dependency management  
FastAPI  
Uvicorn  
LangChain  
LangGraph  
Chainlit  
FAISS (faiss-cpu)  
Google Generative AI  
Tavily API  
python-dotenv  
pydantic  
beautifulsoup4

⚙️ Setup Instructions

Clone the Repository  
git clone https://github.com/masfana016/Telemedicine_BOT.git
cd Telemedicine_BOT/Backend

Install uv  
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv init

Install Dependencies with uv  
uv add requirement_name


Create a .env FileIn the Backend directory, create a .env file with the following:  
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key



🚀 Running the Application

Start the FastAPI Backend  
uvicorn main:app --port 8001 --reload


Launch the Chainlit UIChainlit provides a friendly web interface to interact with the AI assistant:  
uv run chainlit run main.py --port 8003

After running the command, open your browser and go to http://localhost:8000 to chat with the bot.


📁 Project Structure
Backend/
├── main.py              # FastAPI + Chainlit logic
├── hospitals.txt        # Hospital database (text-based)
├── .env                 # Secret API keys
├── requirements.txt     # Project dependencies
├── chainlit.md          # Instructions shown in Chainlit UI
├── .chainlit/           # Chainlit UI config (optional)
└── README.md

🛠 Tech Stack

FastAPI: High-performance API framework  
Uvicorn: Fast ASGI server  
LangChain & LangGraph: AI orchestration and control flow  
FAISS: Embedding-based memory for semantic search  
Google Generative AI: LLM for answering user queries  
Tavily API: Real-time web search API  
Chainlit: UI for conversational AI applications  
uv: Lightweight Python package/environment manager

🤝 Contributing
Want to contribute? Feel free to fork the repo, suggest improvements, or open issues on the GitHub repository.
📄 License
This project is licensed under the MIT License.
