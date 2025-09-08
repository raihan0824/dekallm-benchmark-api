from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, Optional, List
from datetime import datetime

class BenchmarkMetrics(BaseModel):
    average: float
    maximum: float
    minimum: float
    median: float

class BenchmarkThroughput(BaseModel):
    input_tokens_per_second: float
    output_tokens_per_second: float

class BenchmarkMetricsData(BaseModel):
    time_to_first_token: BenchmarkMetrics
    end_to_end_latency: BenchmarkMetrics
    inter_token_latency: BenchmarkMetrics
    token_speed: BenchmarkMetrics
    throughput: BenchmarkThroughput

class BenchmarkConfiguration(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    user: int
    spawnrate: int
    model: str
    tokenizer: str
    url: str
    duration: int

class BenchmarkResults(BaseModel):
    status: str
    metrics: BenchmarkMetricsData
    configuration: BenchmarkConfiguration

class BenchmarkRequest(BaseModel):
    user: Optional[int] = 100
    spawnrate: Optional[int] = 100
    model: Optional[str] = None
    tokenizer: Optional[str] = None
    url: Optional[str] = "https://dekallm.cloudeka.ai"
    duration: Optional[int] = 60
    dataset: Optional[str] = "mteb/banking77"

class BenchmarkResponse(BaseModel):
    id: int
    url: str
    user: int
    spawnrate: int
    duration: int
    model: str
    tokenizer: str
    dataset: str
    notes: Optional[str]
    status: str
    results: BenchmarkResults
    createdAt: datetime

class BenchmarkListResponse(BaseModel):
    results: List[BenchmarkResponse]
    total: int
    page: int
    limit: int

# Data Transfer Object for internal use (moved from models folder)
class BenchmarkResult:
    def __init__(
        self,
        url: str,
        users: int,
        spawn_rate: int,
        duration: int,
        model: str,
        tokenizer: str,
        dataset: str,
        status: str,
        results: Dict[str, Any],
        notes: Optional[str],
        id: Optional[int] = None,
        created_at: Optional[datetime] = None
    ):
        self.id = id
        self.url = url
        self.users = users
        self.spawn_rate = spawn_rate
        self.duration = duration
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.notes = notes
        self.status = status
        self.results = results
        self.created_at = created_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "url": self.url,
            "user": self.users,
            "spawnrate": self.spawn_rate,
            "duration": self.duration,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "dataset": self.dataset,
            "notes": self.notes if self.notes is not None else "",
            "status": self.status,
            "results": self.results,
            "createdAt": self.created_at.isoformat() if self.created_at else None
        }

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> 'BenchmarkResult':
        """Create instance from database row"""
        return cls(
            id=row['id'],
            url=row['url'],
            users=row['users'],
            spawn_rate=row['spawn_rate'],
            duration=row['duration'],
            model=row['model'],
            tokenizer=row['tokenizer'],
            dataset=row['dataset'],
            notes=row['notes'],
            status=row['status'],
            results=row['results'],
            created_at=row['created_at']
        )