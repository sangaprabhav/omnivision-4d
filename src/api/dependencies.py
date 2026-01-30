"""
Dependencies: Auth, rate limiting, etc.
"""
from fastapi import HTTPException, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from src.config import settings
import time
from typing import Dict
import hashlib

security = HTTPBearer()

# Simple rate limiting (in-memory, use Redis in production)
class RateLimiter:
    def __init__(self, requests_per_minute: int = 10):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}  # {api_key: [timestamps]}
    
    def is_allowed(self, api_key: str) -> bool:
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        if api_key not in self.requests:
            self.requests[api_key] = []
        
        # Remove old requests
        self.requests[api_key] = [t for t in self.requests[api_key] if t > window_start]
        
        if len(self.requests[api_key]) >= self.requests_per_minute:
            return False
        
        self.requests[api_key].append(now)
        return True

rate_limiter = RateLimiter(requests_per_minute=10)

async def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Header(None, alias="Authorization")
):
    """
    Verify API key from Authorization header
    Expected format: "Bearer your-api-key"
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing API key")
    
    # Extract token from "Bearer XXX"
    token = credentials.replace("Bearer ", "") if credentials.startswith("Bearer ") else credentials
    
    if token != settings.API_KEY:
        # Log failed attempt
        client_ip = request.client.host if request.client else "unknown"
        print(f"Invalid API key attempt from {client_ip}")
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Rate limiting
    if not rate_limiter.is_allowed(token):
        raise HTTPException(status_code=429, detail="Rate limit exceeded (10 req/min)")
    
    return token

def get_upload_size_limit():
    """Return max upload size in bytes"""
    return settings.MAX_UPLOAD_SIZE