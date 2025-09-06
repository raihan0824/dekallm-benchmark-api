from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from server.routes.benchmark_routes import benchmark_router
from server.database.connection import init_database
import uvicorn
import os
import logging
from contextlib import asynccontextmanager

os.makedirs('logs', exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database on startup
    try:
        init_database()
        logging.getLogger('main').info("Database initialized successfully")
    except Exception as e:
        logging.getLogger('main').error(f"Failed to initialize database: {e}")
        raise
    yield

app = FastAPI(
    title=os.getenv('APP_NAME', 'LLM Benchmark API'),
    description="API for running LLM benchmarks and managing results",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include routers
app.include_router(benchmark_router)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger = logging.getLogger('main')
    logger.error(f"Global exception: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")

@app.get('/')
def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'service': 'LLM Benchmark API',
        'version': '1.0.0'
    }

# 

if __name__ == '__main__':
    # Setup logging here
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logging.getLogger('').addHandler(handler)
    logging.getLogger('').setLevel(logging.INFO)

    logger = logging.getLogger('LLM-API')
    
    logger.info(f"{os.getenv('APP_NAME','my-app')} service started")
    logger.info(f"host: {os.getenv('HOST','0.0.0.0')}:{os.getenv('PORT','8000')}")

    uvicorn.run("main:app", port=int(os.getenv('PORT','8000')), host=os.getenv('HOST','0.0.0.0'),workers=1)
