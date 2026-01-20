import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.exceptions import ApiException, api_exception_handler
from routers import legal

app = FastAPI(
    title=settings.APP_NAME,
    description="法律智能体",
    version="1.0.0"
)

# 开启跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(ApiException, api_exception_handler)

# 配置日志，避免权限问题
logging.basicConfig(
    level=logging.INFO,
    filename = "app.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding = "utf-8"
)

logging.info("法律智能体服务启动成功")

app.include_router(legal.router)

@app.get("/")
async def root():
    return {
        "message": "欢迎使用法律智能体服务",
        "version": "1.0.0",
        "available_agnets": ["legal_agent"]  # 可用智能体
    }


# 启动服务
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
