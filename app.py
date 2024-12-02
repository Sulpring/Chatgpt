from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Debug: Print API key (first few characters)
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key loaded (first 5 chars): {api_key[:5] if api_key else 'None'}")

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": request.message}
            ]
        )
        return ChatResponse(response=response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)