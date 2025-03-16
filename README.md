# Gemini-Proxy

这是一个将OpenAI API请求转发到Google Gemini API的代理服务。它允许使用OpenAI兼容的应用程序无缝地使用Google的Gemini模型。

## 功能特点

- 将OpenAI格式的请求转换为Gemini API格式
- 支持文本和图像输入（多模态）
- 支持流式响应
- 支持系统指令（System Instructions）
- 自动选择合适的模型（图像使用gemini-pro-vision，纯文本使用gemini-pro）
- 详细的错误处理和日志记录

## 安装

1. 克隆此仓库
2. 安装依赖：

```bash
pip install fastapi uvicorn google-generativeai pillow pydantic
```

## 配置

设置环境变量：

```bash
export GEMINI_API_KEY=your_gemini_api_key
```

在Windows上：

```cmd
set GEMINI_API_KEY=your_gemini_api_key
```

## 运行

```bash
python app.py
```

服务将在 http://localhost:11434 上运行。

## API端点

### 聊天补全

```
POST /v1/chat/completions
```

请求格式与OpenAI API兼容，例如：

```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "你好，请介绍一下自己。"}
  ],
  "stream": false
}
```

### 健康检查

```
GET /health
```

## 使用示例

使用curl发送请求：

```bash
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "你好，请告诉我北京的人口大约是多少？"}
    ]
  }'
```

## 限制

- Gemini API不提供token计数，因此使用-1作为占位符
- 某些OpenAI特定功能（如function calling）尚未实现
- 响应格式尽量兼容OpenAI，但可能存在细微差异

## 许可证

[Apache-2.0 license](https://github.com/GuilingQiyu/Gemini-Proxy?tab=Apache-2.0-1-ov-file#)
