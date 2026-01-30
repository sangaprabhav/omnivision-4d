"""
Basic API tests
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_unauthorized_access():
    response = client.post("/jobs", files={"file": ("test.mp4", b"fake video")})
    assert response.status_code == 403  # Should require auth

def test_invalid_file_type():
    # Would need to set up auth header properly for real test
    pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])