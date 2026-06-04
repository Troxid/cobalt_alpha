from typing import Literal

from pydantic import BaseModel, Field

RouterIntent = Literal["direct_answer", "new_computation", "refine_computation"]


class RouterDecision(BaseModel):
    score: float = Field(
        ge=0.0, le=1.0, description="Оцена сложности запроса пользователя"
    )
    intent: RouterIntent = Field(description="Тип пользовательского запроса")


class PlanDraft(BaseModel):
    problem_spec: str = Field(min_length=1)
    plan: str = Field(min_length=1)
    steps: list[str] = Field(default_factory=list)


class PlanVerification(BaseModel):
    approved: bool
    feedback: str = ""
