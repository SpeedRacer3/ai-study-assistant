import os
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
mongo_uri = os.getenv("MONGODB_URI")

# MongoDB Connection
client = MongoClient(mongo_uri)
db = client["study_bot"]
collection = db["chat_history"]

# FastAPI App
app = FastAPI(title="AI Study Assistant")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model
class ChatRequest(BaseModel):
    user_id: str
    question: str

# Study Assistant System Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI Study Assistant. "
            "You answer academic and learning-related questions clearly and simply. "
            "Use previous conversation context when available. "
            "If the question is not study-related, politely guide the user back to academic topics."
        ),
        ("placeholder", "{history}"),
        ("human", "{question}")
    ]
)

# LLM
llm = ChatGroq(
    api_key=groq_api_key,
    model="openai/gpt-oss-20b"
)

chain = prompt | llm


# Function to retrieve chat history
def get_chat_history(user_id: str):
    chats = collection.find({"user_id": user_id}).sort("timestamp", 1)
    history = []

    for chat in chats:
        history.append((chat["role"], chat["message"]))

    return history

@app.get("/")
def home():
    return {"message": "AI Study Assistant API is running!"}


@app.post("/chat")
def chat(request: ChatRequest):
    # Get previous history
    history = get_chat_history(request.user_id)

    # Generate response
    response = chain.invoke({
        "history": history,
        "question": request.question
    })

    # Store user message
    collection.insert_one({
        "user_id": request.user_id,
        "role": "human",
        "message": request.question,
        "timestamp": datetime.utcnow()
    })

    # Store assistant response
    collection.insert_one({
        "user_id": request.user_id,
        "role": "ai",
        "message": response.content,
        "timestamp": datetime.utcnow()
    })

    return {"response": response.content}