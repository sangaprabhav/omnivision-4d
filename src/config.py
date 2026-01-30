import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API
    API_KEY: str = "dev-key-change-in-production"
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500MB
    
    # Models
    MODEL_CACHE_DIR: str = "/app/models"
    SAM2_CHECKPOINT: str = "sam2_hiera_large.pt"
    COSMOS_MODEL: str = "nvidia/Cosmos-Reason2-8B"
    
    # Processing
    MAX_FRAMES: int = 32
    TARGET_FPS: int = 2
    
    # GCP (Your $1K credits)
    GCP_PROJECT: str = ""
    GCS_BUCKET: str = ""
    
    class Config:
        env_file = ".env"

settings = Settings()

