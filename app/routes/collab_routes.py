from fastapi import APIRouter, Query
from typing import List, Dict, Any
from app.utils.pipeline import (
    get_final_context,
    get_answer_from_llm
)

router = APIRouter(
    prefix="/api",
    tags=["collaborative filtering"],
    responses={404: {"description": "Not found"}},
)

@router.get("/answer")
async def get_answer(query: str = Query(...)):
    context = get_final_context(query)
    answer = get_answer_from_llm(query, context)
    return {"answer": answer}