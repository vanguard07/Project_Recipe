from fastapi import FastAPI
from app.api.recipe import router as RecipeRouter
from app.api.chat import router as ChatRouter # Ensure chat router is imported
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings # Import settings if needed elsewhere

app = FastAPI(
    title="RecipeGPT Backend",
    description="NLP-based Recipe Chatbot API",
    version="0.1.0"
)

# Include routers
app.include_router(RecipeRouter, prefix="/recipe", tags=["recipes"])
app.include_router(ChatRouter, prefix="/chat", tags=["chat"]) # Ensure chat router is included

# CORS Middleware
origins = [
    "http://localhost:3000", # Allow frontend origin
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to RecipeGPT API!"}