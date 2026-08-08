from pydantic_settings import BaseSettings, SettingsConfigDict

# read env and validateds as python objects

class Settings(BaseSettings):
    vllm_base_url: str = 'http://localhost:8000'
    vllm_api_key: str ='local-token'
    model_name: str = 'Qwen/Qwen3-1.7B'
    gateway_port: int = 8080
    request_timeout_s: int = 180
    #pydantic read .env 
    model_config = SettingsConfigDict(
        env_file = '.env',
        env_file_encoding = 'utf-8',
        extra = 'ignore',
    )
settings = Settings()
    
    