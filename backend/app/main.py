from fastapi import FastAPI
from app.api.recipe import router as RecipeRouter

app = FastAPI(
    title="RecipeGPT Backend",
    description="NLP-based Recipe Chatbot API",
    version="0.1.0"
)

app.include_router(RecipeRouter, prefix="/recipes", tags=["recipes"])

@app.get("/")
async def root():
    return {"message": "Welcome to RecipeGPT API!"}