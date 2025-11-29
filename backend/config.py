import os

class Config:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    # Add other configuration variables here if needed
