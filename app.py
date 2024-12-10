from fastapi import FastAPI, HTTPException
from openai import OpenAI
import base64
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/analyze")
async def analyze_image(file: str, message: str = None):
    try:
        # base64로 전달받은 이미지 문자열을 바로 사용
        base64_image = file
        
        # 시스템 메시지 설정
        if message:
            system_message = """당신은 한국어로 응답하는 이미지 분석 어시스턴트입니다. 
            사용자의 메시지를 고려하여 이미지를 분석하고, 이미지의 내용과 사용자의 질문/설명을 
            자연스럽게 연결하여 응답해주세요."""
            user_message = f"이미지에 대해 다음 내용을 고려하여 설명해주세요: {message}"
        else:
            system_message = """당신은 한국어로 응답하는 이미지 분석 어시스턴트입니다. 
            이미지의 내용을 자세히 분석하여 자연스러운 한국어로 설명해주세요."""
            user_message = "이 이미지에 대해 설명해주세요."
        
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
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500
        )
        
        return {"response": response.choices[0].message.content}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)