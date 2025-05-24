from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Smart Music API"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Thêm các cấu hình khác ở đây, ví dụ:
    # database_url: str = "postgresql://user:password@localhost/dbname"
    # api_keys: dict = {}
    # cors_origins: list = ["http://localhost:3000"]
    
    class Config:
        env_file = ".env"

settings = Settings()