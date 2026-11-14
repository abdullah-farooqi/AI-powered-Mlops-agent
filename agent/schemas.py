from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

class ToolExecutionResult(BaseModel):
    tool_name: str
    status: str  # "success", "error", "denied"
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_seconds: float
    request_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class AgentStructuredResponse(BaseModel):
    query: str
    known_facts: List[str]
    uncertainties: List[str]
    recommended_actions: List[str]
    tool_results: List[ToolExecutionResult]
    final_narrative: str
