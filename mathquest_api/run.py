"""
Entry point — jalankan server dengan:
    python run.py
atau langsung dengan uvicorn:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import uvicorn
from MathQuestAPI.app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False, 
        workers=1,      
        log_level="info",
    )
