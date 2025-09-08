from typing import List, Optional
from server.database.connection import get_db_connection
from server.schemas.benchmark import BenchmarkResult
import json
import logging

logger = logging.getLogger(__name__)

class BenchmarkService:
    
    @staticmethod
    def create_benchmark(
        url: str,
        users: int,
        spawn_rate: int,
        duration: int,
        model: str,
        tokenizer: str,
        dataset: str,
        notes: str,
        status: str,
        results: dict
    ) -> BenchmarkResult:
        """Create a new benchmark result in database"""
        try:
            insert_query = """
            INSERT INTO benchmark_results (url, users, spawn_rate, duration, model, tokenizer, dataset, notes, status, results)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, created_at;
            """
            
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(insert_query, (
                        url, users, spawn_rate, duration, model, tokenizer, dataset, notes, status, json.dumps(results)
                    ))
                    row = cursor.fetchone()
                    
                    return BenchmarkResult(
                        id=row['id'],
                        url=url,
                        users=users,
                        spawn_rate=spawn_rate,
                        duration=duration,
                        model=model,
                        tokenizer=tokenizer,
                        dataset=dataset,
                        notes=notes,
                        status=status,
                        results=results,
                        created_at=row['created_at']
                    )
        except Exception as e:
            logger.error(f"Error creating benchmark: {e}")
            raise

    @staticmethod
    def get_benchmark_by_id(benchmark_id: int) -> Optional[BenchmarkResult]:
        """Get benchmark result by ID"""
        try:
            select_query = """
            SELECT id, url, users, spawn_rate, duration, model, tokenizer, dataset, notes, status, results, created_at
            FROM benchmark_results
            WHERE id = %s;
            """
            
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(select_query, (benchmark_id,))
                    row = cursor.fetchone()
                    
                    if row:
                        return BenchmarkResult.from_db_row(row)
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting benchmark by ID: {e}")
            raise

    @staticmethod
    def get_all_benchmarks(page: int = 1, limit: int = 10) -> tuple[List[BenchmarkResult], int]:
        """Get all benchmark results with pagination"""
        try:
            offset = (page - 1) * limit
            
            # Get total count
            count_query = "SELECT COUNT(*) FROM benchmark_results;"
            
            # Get paginated results
            select_query = """
            SELECT id, url, users, spawn_rate, duration, model, tokenizer, dataset, notes, status, results, created_at
            FROM benchmark_results
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s;
            """
            
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Get total count
                    cursor.execute(count_query)
                    total = cursor.fetchone()['count']
                    
                    # Get paginated results
                    cursor.execute(select_query, (limit, offset))
                    rows = cursor.fetchall()
                    
                    benchmarks = [BenchmarkResult.from_db_row(row) for row in rows]
                    return benchmarks, total
                    
        except Exception as e:
            logger.error(f"Error getting all benchmarks: {e}")
            raise

    @staticmethod
    def update_benchmark(
        benchmark_id: int,
        url: Optional[str] = None,
        users: Optional[int] = None,
        spawn_rate: Optional[int] = None,
        duration: Optional[int] = None,
        model: Optional[str] = None,
        tokenizer: Optional[str] = None,
        dataset: Optional[str] = None,
        notes: Optional[str] = None,
        status: Optional[str] = None,
        results: Optional[dict] = None
    ) -> Optional[BenchmarkResult]:
        """Update benchmark result"""
        try:
            # Build dynamic update query
            updates = []
            params = []
            
            if url is not None:
                updates.append("url = %s")
                params.append(url)
            if users is not None:
                updates.append("users = %s")
                params.append(users)
            if spawn_rate is not None:
                updates.append("spawn_rate = %s")
                params.append(spawn_rate)
            if duration is not None:
                updates.append("duration = %s")
                params.append(duration)
            if model is not None:
                updates.append("model = %s")
                params.append(model)
            if tokenizer is not None:
                updates.append("tokenizer = %s")
                params.append(tokenizer)
            if dataset is not None:
                updates.append("dataset = %s")
                params.append(dataset)
            if notes is not None:
                updates.append("notes = %s")
                params.append(notes)
            if status is not None:
                updates.append("status = %s")
                params.append(status)
            if results is not None:
                updates.append("results = %s")
                params.append(json.dumps(results))
            
            if not updates:
                return BenchmarkService.get_benchmark_by_id(benchmark_id)
            
            params.append(benchmark_id)
            update_query = f"""
            UPDATE benchmark_results 
            SET {', '.join(updates)}
            WHERE id = %s
            RETURNING id, url, users, spawn_rate, duration, model, tokenizer, dataset, notes, status, results, created_at;
            """
            
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(update_query, params)
                    row = cursor.fetchone()
                    
                    if row:
                        return BenchmarkResult.from_db_row(row)
                    return None
                    
        except Exception as e:
            logger.error(f"Error updating benchmark: {e}")
            raise

    @staticmethod
    def delete_benchmark(benchmark_id: int) -> bool:
        """Delete benchmark result"""
        try:
            delete_query = "DELETE FROM benchmark_results WHERE id = %s;"
            
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(delete_query, (benchmark_id,))
                    return cursor.rowcount > 0
                    
        except Exception as e:
            logger.error(f"Error deleting benchmark: {e}")
            raise