import os
import time
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Generator, Any
import google.generativeai as genai
from PIL import Image
import requests
import base64
import io
import json
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levellevel)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenAI to Gemini Proxy")

# 配置Gemini客户端
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class OpenAIMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: Union[str, List[Dict[str, Any]]]

class OpenAIRequest(BaseModel):
    model: str = "gpt-3.5-turbo"
    messages: List[OpenAIMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None

def validate_api_key(authorization: Optional[str] = Header(None)):
    """验证API密钥是否存在，支持从请求头获取"""
    api_key = None
    
    # 优先从请求头中获取API密钥
    if authorization:
        if authorization.startswith("Bearer "):
            api_key = authorization.replace("Bearer ", "")
    
    # 如果请求头没有API密钥，则使用环境变量中的默认密钥
    if not api_key:
        api_key = GEMINI_API_KEY
    
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": "API Key not provided", "type": "api_error"}}
        )
    
    # 配置Gemini客户端
    genai.configure(api_key=api_key)
    return api_key

def check_for_images_in_content(messages: List[OpenAIMessage]) -> bool:
    """检查消息中是否包含图像"""
    for msg in messages:
        if msg.role == "user" and isinstance(msg.content, list):
            for item in msg.content:
                if item.get("type") == "image_url":
                    return True
    return False

def get_appropriate_model(model_name: str, has_images: bool) -> str:
    """处理模型选择，仅在有图像时强制使用vision模型"""
    # 如果有图像且模型名称中不包含vision，切换到视觉模型
    if has_images and "vision" not in model_name.lower():
        logger.info(f"请求包含图像，将模型从 {model_name} 切换为 gemini-pro-vision")
        return "gemini-pro-vision"
    
    # 直接使用客户端传入的模型名称
    return model_name

async def fetch_google_models():
    """从Google API获取可用模型列表"""
    try:
        # 确保API密钥已配置
        genai.configure(api_key=GEMINI_API_KEY)
        
        # 获取所有可用模型
        models = genai.list_models()
        return models
    except Exception as e:
        logger.error(f"Error fetching models from Google API: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": f"Failed to fetch models: {str(e)}", "type": "api_error"}}
        )

def convert_messages(messages: List[OpenAIMessage]) -> tuple:
    """将OpenAI消息转换为Gemini格式"""
    system_instruction = ""
    parts = []
    has_images = False
    
    try:
        for msg in messages:
            if msg.role == "system":
                system_instruction += msg.content if isinstance(msg.content, str) else str(msg.content)
                system_instruction += "\n"
            elif msg.role == "user":
                if isinstance(msg.content, str):
                    parts.append(msg.content)
                else:  # 处理多模态内容
                    text_parts = []
                    for content_item in msg.content:
                        if content_item["type"] == "text":
                            text_parts.append(content_item["text"])
                        elif content_item["type"] == "image_url":
                            has_images = True
                            image_url = content_item["image_url"]["url"]
                            try:
                                if image_url.startswith("data:image"):
                                    # Base64解码
                                    header, data = image_url.split(",", 1)
                                    image_bytes = base64.b64decode(data)
                                else:
                                    # 下载图片
                                    response = requests.get(image_url, timeout=10)
                                    response.raise_for_status()
                                    image_bytes = response.content
                                
                                image = Image.open(io.BytesIO(image_bytes))
                                if text_parts:  # 如果有之前的文本，先添加文本
                                    parts.append("\n".join(text_parts))
                                    text_parts = []
                                parts.append(image)
                            except Exception as e:
                                logger.error(f"Error processing image: {str(e)}")
                                raise HTTPException(
                                    status_code=400, 
                                    detail={"error": {"message": f"Failed to process image: {str(e)}", "type": "invalid_request_error"}}
                                )
                    
                    if text_parts:  # 添加剩余的文本
                        parts.append("\n".join(text_parts))
            elif msg.role == "assistant" and isinstance(msg.content, str):
                # 处理对话历史中的助手回复
                parts.append(msg.content)
        
        return system_instruction, parts, has_images
    except Exception as e:
        logger.error(f"Error converting messages: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": f"Failed to convert messages: {str(e)}", "type": "invalid_request_error"}}
        )

def generate_openai_format(content: str, model_name: str, finish_reason: str = None) -> Dict:
    """生成OpenAI兼容的响应格式"""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason or "stop"
        }],
        "usage": {
            "prompt_tokens": -1,  # Gemini不提供token计数
            "completion_tokens": -1,
            "total_tokens": -1
        }
    }

def stream_generator(response, model_name: str) -> Generator[str, None, None]:
    """流式响应生成器"""
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    try:
        for chunk in response:
            if not chunk.text:
                continue
            
            yield f"data: {json.dumps({'id': response_id,'object': 'chat.completion.chunk','created': int(time.time()), 'model': model_name,'choices': [{'index': 0,'delta': {'content': chunk.text},'finish_reason': None}]})}\n\n"
        
        # 发送完成信号
        yield f"data: {json.dumps({ 'id': response_id, 'object': 'chat.completion.chunk','created': int(time.time()),'model': model_name,'choices': [{'index': 0,'delta': {},'finish_reason': 'stop'}] })}\n\n"
        
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error in stream generation: {str(e)}")
        error_json = json.dumps({"error": {"message": str(e), "type": "api_error"}})
        yield f"data: {error_json}\n\n"
        yield "data: [DONE]\n\n"

@app.get("/v1/models")
async def list_models(api_key_valid: bool = Depends(validate_api_key)):
    """返回可用的模型列表，从Google API获取并转换为OpenAI兼容格式"""
    google_models = await fetch_google_models()
    
    models_data = []
    for model in google_models:
        models_data.append({
            "id": model.name,  # 使用模型的原始名称
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google",
            "permission": [],
            "root": model.name,
            "parent": None
        })
    
    return {"object": "list", "data": models_data}

@app.post("/v1/chat/completions")
async def chat_completion(
    request: OpenAIRequest, 
    api_key: str = Depends(validate_api_key)
):
    try:
        logger.info(f"Received request for model: {request.model}")
        
        # 检查是否有图像内容
        has_images = check_for_images_in_content(request.messages)
        
        # 获取适当的模型名称
        model_name = get_appropriate_model(request.model, has_images)
        logger.info(f"Using model: {model_name}")
        
        # 转换消息和参数
        system_instruction, parts, _ = convert_messages(request.messages)
        
        # 检查是否有内容
        if not parts:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": "No valid content found in messages", "type": "invalid_request_error"}}
            )
        
        # 准备生成配置
        generation_config = {}
        if request.max_tokens is not None:
            generation_config["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.top_p is not None:
            generation_config["top_p"] = request.top_p
        
        # 添加频率惩罚和存在惩罚参数
        # Gemini API将频率惩罚映射为frequency_penalty_scale
        if request.frequency_penalty is not None:
            generation_config["frequency_penalty"] = request.frequency_penalty
        
        # Gemini API将存在惩罚映射为presence_penalty_scale
        if request.presence_penalty is not None:
            generation_config["presence_penalty"] = request.presence_penalty
            
        if request.stop is not None:
            generation_config["stop_sequences"] = request.stop
        
        # 初始化模型
        model = genai.GenerativeModel(
            model_name,
            system_instruction=system_instruction if system_instruction else None,
        )

        # 流式处理
        if request.stream:
            try:
                response = model.generate_content(
                    parts,
                    stream=True,
                    generation_config=genai.types.GenerationConfig(**generation_config) if generation_config else None
                )
                return StreamingResponse(
                    stream_generator(response, model_name),
                    media_type="text/event-stream"
                )
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail={"error": {"message": f"Streaming error: {str(e)}", "type": "api_error"}}
                )

        # 普通响应
        try:
            response = model.generate_content(
                parts,
                generation_config=genai.types.GenerationConfig(**generation_config) if generation_config else None
            )
            return generate_openai_format(response.text, model_name)
        except genai.types.BlockedPromptException as e:
            logger.warning(f"Blocked prompt: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": "Your prompt was blocked for safety reasons.", "type": "content_filter"}}
            )
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={"error": {"message": str(e), "type": "api_error"}}
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"An unexpected error occurred: {str(e)}",
                    "type": "api_error"
                }
            }
        )

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok"}

if __name__ == "__main__":
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set. Please set it before making requests.")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=11434,
        log_level="info",
        timeout_keep_alive=30
    )