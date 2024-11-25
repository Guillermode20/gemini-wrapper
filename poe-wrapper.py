import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from gemini_webapi import GeminiClient
from datetime import datetime
import time

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "gpt-3.5-turbo"  # Changed to match OpenAI format
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage

class GeminiService:
    def __init__(self):
        self.Secure_1PSID = ""
        self.Secure_1PSIDTS = ""
        self.client = None

    async def init_client(self):
        if not self.client:
            self.client = GeminiClient(self.Secure_1PSID, self.Secure_1PSIDTS, proxies=None)
            await self.client.init(timeout=30, auto_close=False, close_delay=300, auto_refresh=True)

    async def estimate_tokens(self, text: str) -> int:
        # Rough estimation: 4 characters per token
        return len(text) // 4

    async def get_response(self, messages: List[ChatMessage], temperature: float = 0.7) -> tuple[str, int, int]:
        prompt = " ".join([msg.content for msg in messages])
        
        # Remove all parameters - gemini-webapi doesn't support them directly
        response = await self.client.generate_content(prompt)
        
        prompt_tokens = await self.estimate_tokens(prompt)
        completion_tokens = await self.estimate_tokens(response.text)
        
        return response.text, prompt_tokens, completion_tokens

gemini_service = GeminiService()

@app.on_event("startup")
async def startup_event():
    await gemini_service.init_client()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Incoming request:")
        print("User:", request.messages[-1].content)

        # Pass temperature from request
        response_text, prompt_tokens, completion_tokens = await gemini_service.get_response(
            request.messages,
            temperature=request.temperature
        )
        
        print(f"Assistant: {response_text}\n")
        print("-" * 80)
        
        return ChatResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response_text
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)