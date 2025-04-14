from fastapi import APIRouter

process_router = APIRouter(prefix='/process')

@process_router.get('/')
def testing():
    return {"message": "hello there everyone"}
