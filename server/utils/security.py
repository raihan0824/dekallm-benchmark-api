import os
import logging
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

if os.getenv('BASIC_AUTH', 'False') == 'True':
    logger.warning('API is running using HTTP Basic credentials')
    security = HTTPBasic()
    async def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
        correct_username = os.getenv('API_USERNAME', 'username')
        correct_password = os.getenv('API_PASSWORD', 'password')
        if credentials.username != correct_username or credentials.password != correct_password:
            raise HTTPException(
                status_code=401,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Basic"},
            )
        return credentials.username
else:
    logger.warning('API is running without HTTP Basic credentials')
    async def get_current_user():
        return ''