from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class AskIn(BaseModel):
    q: str

class AskOut(BaseModel):
    intent: str
    question: str
    result: Optional[Dict[str, Any]] = None
    need: Optional[List[str]] = None
    message: str
    error: Optional[str] = None
