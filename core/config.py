from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Redis配置
    REDIS_HOST: str = "47.110.65.11"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = "190887lwj"
    REDIS_MAX_CONNECTIONS: int = 10

    # MySQL配置
    MYSQL_HOST: str = "47.110.65.11"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "190887"
    MYSQL_DATABASE: str = "Multi"
    MYSQL_CHARSET: str = "utf8mb4"

    # 应用配置
    APP_NAME: str = "多模态智能体中心"
    DEBUG: bool = False

    # JWT配置
    JWT_SECRET_KEY: str = "KvEZBie00uAnrqhZxUH0VuDWvDzlwVWSnJGCsC+A/ww="
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_LOGIN_SUBJECT: str = "MAOLIUSU"

    # Web SEARCHAPI_API_KEY
    SEARCHAPI_API_KEY: str = "qBiL44ufwBNRWkWogWhs3ewc"

    # LLM配置 阿里云百炼统一接入配置
    # 千问配置
    LLM_MODEL_NAME: str = "qwen-plus"
    LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_API_KEY: str = "sk-fefb71cca5304a128895ccb3e87d6add"
    LLM_TEMPERATURE: float = 0.7
    LLM_STREAMING: bool = True

    # 多模态配置
    # Qwen(图片)
    QWEN_MODEL_NAME: str = "qwen3-vl-plus"
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_API_KEY: str = "sk-fefb71cca5304a128895ccb3e87d6add"
    QWEN_TEMPERATURE: float = 0.1
    QWEN_STREAMING: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
