import logging
import uvicorn

from fastapi import FastAPI, HTTPException
from create import CodeGenerator, CodeRequest


app = FastAPI(
    title="Python Code Generator API (LangChain)",
    description="LangChain + Gemini 기반 Python 코드 생성 API",
    version="2.0.0"
)


@app.post("/generate-code")
async def generate_code_api(request: CodeRequest):
    """LangChain 기반 비동기 코드 생성 API"""
    try:
        generated_code = await CodeGenerator.generate_code(request)
        return {"generated_code": generated_code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)