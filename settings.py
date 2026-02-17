from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GROWW_API_KEY: str
    GROWW_API_SECRET: str
    DATABASE_URL: str
    LOG_LEVEL: str = "INFO"
    TELEGRAM_BOT_TOKEN: str = None
    TELEGRAM_CHAT_ID: str = None

    class Config:
        env_file = ".env"

settings = Settings()
