# 一个单纯的代理服务器，用于将请求转发到Google AI API的服务器上
# 如果你只是想单纯地代理请求，可以使用这个脚本

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import logging
import uvicorn
from typing import Dict, Any, AsyncGenerator

# 配置日志 - 增加更详细的日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Google AI API 代理服务器")

# 添加CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TARGET_URL = "https://generativelanguage.googleapis.com"

@app.api_route('/{path:path}', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
async def proxy(request: Request, path: str = ""):
    logger.info(f"接收到请求: {request.method} {request.url.path}")
    
    # 构建目标URL
    target_url = f"{TARGET_URL}/{path}"
    
    # 获取所有请求头，但替换或移除部分头信息
    headers = dict(request.headers)
    
    # 移除Host头，因为httpx会自动添加正确的Host
    if 'host' in headers:
        del headers['host']
    
    # 如果原始请求包含客户端IP，则保留它
    client_host = request.client.host if request.client else None
    if 'x-forwarded-for' not in headers and client_host:
        headers['x-forwarded-for'] = client_host
    
    # 准备转发请求的参数
    kwargs: Dict[str, Any] = {
        'headers': headers,
        'follow_redirects': False,
    }
    
    # 处理查询参数
    params = dict(request.query_params)
    if params:
        kwargs['params'] = params
    
    # 处理请求体
    content_type = request.headers.get('content-type', '')
    
    try:
        if 'application/json' in content_type:
            kwargs['json'] = await request.json()
        elif 'application/x-www-form-urlencoded' in content_type:
            kwargs['data'] = await request.form()
        else:
            body = await request.body()
            if body:
                kwargs['content'] = body
    except Exception as e:
        logger.error(f"解析请求体时出错: {e}")
        return Response(content=f"请求体解析错误: {str(e)}", status_code=400)
    
    # 创建 httpx 客户端进行请求
    async with httpx.AsyncClient(timeout=120.0) as client:  # 增加超时时间至2分钟
        try:
            # 发送请求到目标服务器
            resp = await client.request(
                method=request.method,
                url=target_url,
                **kwargs
            )
            
            logger.info(f"从API收到响应: 状态码 {resp.status_code}")
            
            # 准备响应头
            response_headers: Dict[str, str] = dict(resp.headers)
            
            # 移除可能导致问题的头部
            problematic_headers = ['content-encoding', 'content-length', 'transfer-encoding']
            for header in problematic_headers:
                if header in response_headers:
                    del response_headers[header]
            
            # 确保设置正确的内容类型
            if 'content-type' in resp.headers:
                response_headers['content-type'] = resp.headers['content-type']
            
            # 创建更高效的流式响应
            async def generate() -> AsyncGenerator[bytes, None]:
                received_chunks = 0
                try:
                    # 使用稍大一点的chunk_size以减少网络开销，但保持较好的实时性
                    async for chunk in resp.aiter_bytes(chunk_size=512):
                        received_chunks += 1
                        if received_chunks % 50 == 0:  # 每50个数据块记录一次日志，避免日志过多
                            logger.debug(f"正在流式传输数据: 已传输{received_chunks}个数据块")
                        yield chunk
                    logger.info(f"流式传输完成: 总共传输{received_chunks}个数据块")
                except Exception as e:
                    logger.error(f"流式响应错误: {e}")
                    yield f"流式响应错误: {str(e)}".encode()
            
            # 返回响应
            return StreamingResponse(
                generate(),
                status_code=resp.status_code,
                headers=response_headers,
                media_type=resp.headers.get('content-type')
            )
        
        except httpx.RequestError as e:
            logger.error(f"代理请求错误: {e}")
            return Response(content=f"代理错误: {str(e)}", status_code=500)

if __name__ == '__main__':
    print(f"代理服务器启动在 http://0.0.0.0:11434")
    print(f"正在代理 {TARGET_URL}")
    uvicorn.run("app:app", host="0.0.0.0", port=11434, reload=True)