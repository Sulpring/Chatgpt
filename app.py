from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from openai import OpenAI
import base64
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ImageAnalysisRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/analyze", response_model=ChatResponse)
async def analyze_image(
    file: UploadFile = File(...),
    message: Optional[str] = Form(default=None)
):
    try:
        # Read the image file
        image_content = await file.read()
        
        # Encode the image to base64
        base64_image = base64.b64encode(image_content).decode('utf-8')
        
        # Prepare the system message based on whether there's user input
        if message:
            system_message = """당신은 한국어로 응답하는 이미지 분석 어시스턴트입니다. 
            사용자의 메시지를 고려하여 이미지를 분석하고, 이미지의 내용과 사용자의 질문/설명을 
            자연스럽게 연결하여 응답해주세요."""
            user_message = f"이미지에 대해 다음 내용을 고려하여 설명해주세요: {message}"
        else:
            system_message = """당신은 한국어로 응답하는 이미지 분석 어시스턴트입니다. 
            이미지의 내용을 자세히 분석하여 자연스러운 한국어로 설명해주세요."""
            user_message = "이 이미지에 대해 설명해주세요."
        
        # Create the messages with the image and Korean language instruction
        messages = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{file.content_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        # Make the API call
        response = client.chat.completions.create(
            model="gpt-4o",  # Updated to use the vision model
            messages=messages,
            max_tokens=500
        )
        
        return ChatResponse(response=response.choices[0].message.content)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ImageAnalysisRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using standard GPT-4 for text-only chat
            messages=[
                {"role": "system", "content": "당신은 한국어로 응답하는 AI 어시스턴트입니다. 모든 대화를 한국어로 진행해 주세요."},
                {"role": "user", "content": request.message}
            ]
        )
        return ChatResponse(response=response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)