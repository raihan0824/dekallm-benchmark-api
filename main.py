import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.routes import main_services
import uvicorn
import os
import logging
os.makedirs('logs', exist_ok=True)

app = FastAPI(
    title=os.getenv('APP_NAME','my-app'),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(main_services.generation_router)
app.include_router(main_services.general_router)

@app.get('/')
def ping():
    return{'msg':'acknowledged'}

# 

if __name__ == '__main__':
    # Setup logging here
    class FileAndConsoleHandler(logging.StreamHandler):
        def __init__(self, filename, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.file_handler = logging.FileHandler(filename)

        def emit(self, record):
            # Write log to the console
            super().emit(record)
            
            # Write log to the file
            self.file_handler.emit(record)

        def close(self):
            self.file_handler.close()

    handler = FileAndConsoleHandler(f'./logs/{os.getenv('APP_NAME','my-app')}.log')
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logging.getLogger('').addHandler(handler)
    logging.getLogger('').setLevel(logging.INFO)

    logger = logging.getLogger('LLM-API')
    
    logger.info(f"{os.getenv('APP_NAME','my-app')} service started")
    logger.info(f"host: {os.getenv('HOST','0.0.0.0')}:{os.getenv('PORT','8000')}")

    uvicorn.run("main:app", port=int(os.getenv('PORT','8000')), host=os.getenv('HOST','0.0.0.0'),workers=1)
