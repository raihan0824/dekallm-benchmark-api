import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# Database connection parameters
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "benchmark_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password"),
    "port": os.getenv("DB_PORT", "5432")
}

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def init_database():
    """Initialize database tables"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS benchmark_results (
        id SERIAL PRIMARY KEY,
        url VARCHAR(255) NOT NULL,
        users INTEGER NOT NULL,
        spawn_rate INTEGER NOT NULL,
        duration INTEGER NOT NULL,
        model VARCHAR(255) NOT NULL,
        tokenizer VARCHAR(255) NOT NULL,
        dataset VARCHAR(255) NOT NULL DEFAULT 'mteb/banking77',
        favorite BOOLEAN NOT NULL DEFAULT FALSE,
        notes VARCHAR(255) NOT NULL,
        status VARCHAR(100) NOT NULL,
        results JSONB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise