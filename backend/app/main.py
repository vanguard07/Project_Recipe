from fastapi import FastAPI
from app.api.recipe import router as RecipeRouter
from app.api.chat import router as ChatRouter
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="RecipeGPT Backend",
    description="NLP-based Recipe Chatbot API",
    version="0.1.0"
)

app.include_router(RecipeRouter, prefix="/recipe", tags=["recipes"])
app.include_router(ChatRouter, prefix="/chat", tags=["chat"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to RecipeGPT API!"}