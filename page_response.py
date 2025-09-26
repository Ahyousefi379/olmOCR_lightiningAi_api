from pydantic import BaseModel, Field
from typing import Optional

class PageResponse(BaseModel):
    natural_text: str = Field(..., description="Extracted natural language text from the page")
    primary_language: Optional[str] = Field(None, description="Detected primary language of the page")
    is_table: bool = Field(False, description="True if the page mainly contains a table")
    is_diagram: bool = Field(False, description="True if the page mainly contains a diagram")
    is_rotation_valid: bool = Field(True, description="Whether rotation is correct")
    rotation_correction: int = Field(0, description="Suggested rotation correction (degrees)")
