from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional
import logging
from server.schemas.benchmark import (
    BenchmarkRequest, BenchmarkResponse, BenchmarkListResponse
)
from server.services.benchmark_service import BenchmarkService
from server.services.benchmark_runner import BenchmarkRunner
from server.utils.security import get_current_user

logger = logging.getLogger(__name__)

benchmark_router = APIRouter(
    prefix="/api/v1/benchmarks",
    tags=["Benchmarks"],
    dependencies=[Depends(get_current_user)]
)

@benchmark_router.post("/run", response_model=BenchmarkResponse)
async def run_benchmark_test(
    user: Optional[int] = Query(default=100, description="Number of concurrent users"),
    spawnrate: Optional[int] = Query(default=100, description="User spawn rate per second"),
    model: Optional[str] = Query(default=None, description="Model name to benchmark"),
    tokenizer: Optional[str] = Query(default=None, description="Tokenizer name"),
    url: Optional[str] = Query(default="https://dekallm.cloudeka.ai", description="Target API URL"),
    duration: Optional[int] = Query(default=60, description="Test duration in seconds"),
    dataset: Optional[str] = Query(default="mteb/banking77", description="Dataset for benchmark prompts")
):
    """
    Run a benchmark test and save results to database
    """
    try:
        logger.info(f"Starting benchmark test with params: users={user}, spawn_rate={spawnrate}, duration={duration}")
        
        # Run the benchmark
        results = BenchmarkRunner.run_benchmark(
            users=user,
            spawn_rate=spawnrate,
            duration=duration,
            url=url,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset
        )
        
        # Save to database
        benchmark_result = BenchmarkService.create_benchmark(
            url=url,
            users=user,
            spawn_rate=spawnrate,
            duration=duration,
            model=results["configuration"]["model"],
            tokenizer=results["configuration"]["tokenizer"],
            dataset=results["configuration"]["dataset"],
            status=results["status"],
            results=results
        )
        
        logger.info(f"Benchmark completed and saved with ID: {benchmark_result.id}")
        return benchmark_result.to_dict()
        
    except Exception as e:
        logger.error(f"Error in run_benchmark_test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@benchmark_router.get("/", response_model=BenchmarkListResponse)
async def get_all_benchmarks(
    page: int = Query(default=1, ge=1, description="Page number"),
    limit: int = Query(default=10, ge=1, le=100, description="Number of results per page")
):
    """
    Get all benchmark results with pagination
    """
    try:
        benchmarks, total = BenchmarkService.get_all_benchmarks(page=page, limit=limit)
        
        return {
            "results": [benchmark.to_dict() for benchmark in benchmarks],
            "total": total,
            "page": page,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error getting benchmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@benchmark_router.get("/{benchmark_id}", response_model=BenchmarkResponse)
async def get_benchmark_by_id(benchmark_id: int):
    """
    Get specific benchmark result by ID
    """
    try:
        benchmark = BenchmarkService.get_benchmark_by_id(benchmark_id)
        
        if not benchmark:
            raise HTTPException(status_code=404, detail="Benchmark not found")
            
        return benchmark.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting benchmark by ID: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@benchmark_router.put("/{benchmark_id}", response_model=BenchmarkResponse)
async def update_benchmark(
    benchmark_id: int,
    url: Optional[str] = None,
    users: Optional[int] = None,
    spawn_rate: Optional[int] = None,
    duration: Optional[int] = None,
    model: Optional[str] = None,
    tokenizer: Optional[str] = None,
    dataset: Optional[str] = None,
    status: Optional[str] = None
):
    """
    Update benchmark result
    """
    try:
        benchmark = BenchmarkService.update_benchmark(
            benchmark_id=benchmark_id,
            url=url,
            users=users,
            spawn_rate=spawn_rate,
            duration=duration,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            status=status
        )
        
        if not benchmark:
            raise HTTPException(status_code=404, detail="Benchmark not found")
            
        return benchmark.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating benchmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@benchmark_router.delete("/{benchmark_id}")
async def delete_benchmark(benchmark_id: int):
    """
    Delete benchmark result
    """
    try:
        success = BenchmarkService.delete_benchmark(benchmark_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Benchmark not found")
            
        return {"message": "Benchmark deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting benchmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))