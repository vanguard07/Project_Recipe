# RecipeGPT

RecipeGPT is a conversational recipe assistant powered by GPT and LangChain. It allows users to search, customize, and generate cooking recipes in natural language using either GPT-based workflows or a Retrieval-Augmented Generation (RAG) approach. It supports ingredient-based filtering, nutritional queries, and live customization of recipes.

Demo video - https://youtu.be/U6_SnZBQbIA

---

## Overview
RecipeGPT is a full-stack application that enables users to:
- Search Recipes with Natural Language: Ask things like “Show me a high-protein dinner under 30 minutes”.
- Customize Existing Recipes: Modify ingredients, cooking time, or dietary preferences on the fly.
- Chat-Based Interaction
    Two modes available:
        - OpenAI GPT API-based assistant
        - LangChain RAG-based assistant with ChromaDB vector search
- Context-Aware Responses: Incorporates multi-turn memory and conversation history.
- Fallback Logic: 
    If no recipe matches, the system:
        - Generates a custom recipe via GPT
        - Stores it for future search

The system leverages OpenAI's GPT models, LangChain, MongoDB, and ChromaDB for intelligent recipe extraction, search, and conversational experiences.

---

## Architecture & Workflow

### Frontend:
- Built with Next.js and React.
- Provides UI for recipe management, chat, and search.
- Communicates with backend via REST API.

### Backend:
- FastAPI server exposes endpoints for recipe extraction, search, chat, and customization.
- Integrates with OpenAI (GPT-4, GPT-4o, o3-mini) for NLP tasks.
- Uses LangChain for retrieval-augmented generation (RAG) and context-aware chat.
- Stores recipes and chat history in MongoDB.
- Uses ChromaDB as a vector store for semantic search and retrieval.

#### GPT-based Flow
- /classify: Classifies prompt into search/customization
- /search: Filters and queries recipe_collection based on user input
- /customize: Tailors an existing recipe using past chat history
- MongoDB used to store:
    - Recipes
    - Chat history per user
![Recipe Chatbot Process Guide](https://github.com/user-attachments/assets/6d5df1b8-11eb-4476-b816-2b99f45f9f78)


#### RAG-based Flow (LangChain)
- Vector store: ChromaDB
- Embeddings: OpenAIEmbeddings
- Retriever: Finds similar recipes based on semantic match
- LLM: GPT-4.1 via OpenAI API
- Chain: ConversationalRetrievalChain for contextual, document-grounded answers
![ChatGPT Image May 5, 2025 at 02_36_58 AM](https://github.com/user-attachments/assets/c9f6d294-6ac8-4910-a998-4bdc3a1ce485)

---

## Technologies Used
- **Frontend:** Next.js, React, Tailwind CSS, Lucide Icons
- **Backend:** FastAPI, Python, Pydantic
- **NLP & AI:** OpenAI GPT-4/4o/o3-mini, LangChain
- **Database:** MongoDB (for recipes and chat history)
- **Vector Store:** ChromaDB (for semantic search)
- **Other:** Docker (for containerization), Radix UI, Recharts

---

## Setup & Installation

### Prerequisites
- Docker and Docker Compose
- OpenAI API key

### Quick Start with Docker

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Project_Recipe
   ```

2. **Configure environment variables:**
   ```bash
   # Create a .env file in the root directory
   echo "OPENAI_API_KEY=your-api-key" > .env
   echo "MONGODB_URI=mongodb://mongodb:27017/recipes" >> .env
   echo "CHROMA_PERSIST_DIR=/data/chroma" >> .env
   ```

3. **Build and run the containers:**
   ```bash
   docker-compose up -d
   ```

4. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

5. **Stop the application:**
   ```bash
   docker-compose down
   ```

### Container Details
- **Frontend Container**: Next.js application accessible on port 3000
- **Backend Container**: FastAPI server accessible on port 8000
- **MongoDB Container**: Database service for recipe storage
- **ChromaDB Container**: Vector database for semantic search capabilities

---

## Usage

- **Extract a Recipe:** Paste a recipe URL in the UI and click "Extract Recipe". The recipe will be parsed and added to your collection.
- **Search Recipes:** Use natural language queries (e.g., "Show me vegan pasta recipes under 30 minutes") in the chat or search bar.
- **Chat with AI:** Use the chat interface in one of two modes:
  - **OpenAI GPT Mode**: For general recipe questions, cooking advice, and standard customizations.
  - **LangChain RAG Mode**: For personalized suggestions based on your recipe collection, with the ability to reference and combine ideas from your stored recipes.
- **Manage Recipes:** View, delete, and explore your saved recipes with detailed ingredients, instructions, and nutrition info.

---

## API Endpoints (Backend)
- `POST /recipe/store` — Extract and store a recipe from a URL.
- `GET /recipe/` — List stored recipes.
- `POST /recipe/search` — Search recipes with a natural language prompt.
- `POST /recipe/customize` — Request recipe customization via chat.
- `POST /recipe/classify` — Classify user prompt (search/customize/other).
- `POST /recipe/langchain` — LangChain-powered RAG chat endpoint.

---

## Example Workflow
1. User pastes a recipe URL and clicks "Extract Recipe".
2. Backend fetches and parses the recipe using OpenAI, stores it in MongoDB and ChromaDB.
3. User searches or chats about recipes; backend uses AI and vector search to find and format results.
4. User can request customizations or generate new recipes via chat.

---

## Team Members:
- Avtans Kumar - AXK220317
- Anusha Gupta - AXG230026
