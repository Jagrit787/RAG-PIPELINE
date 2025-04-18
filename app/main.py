from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import collab_routes

app = FastAPI(
    title="RAG API",
    description="A FastAPI application for RAG (Retrieval-Augmented Generation)",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(collab_routes.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 