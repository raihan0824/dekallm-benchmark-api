# LLM Benchmark API

A FastAPI-based service for running LLM benchmarks using Locust and storing results in PostgreSQL.

## Features

- 🚀 **LLM Benchmarking**: Run performance tests against LLM APIs
- 📊 **Comprehensive Metrics**: TTFT, E2E latency, throughput, token speed
- 🗄️ **Database Storage**: PostgreSQL with CRUD operations
- 🔐 **Authentication**: HTTP Basic Auth support
- 📝 **API Documentation**: Auto-generated OpenAPI docs
- 🐳 **Docker Support**: Ready for containerization

## Project Structure

```
dekallm-benchmark-api/
├── main.py                         # FastAPI application entry point
├── requirements.txt                # Python dependencies
├── docker-compose.yaml            # Docker configuration
├── .env.example                   # Environment variables template
├── locustfile.py                  # Backwards compatibility
└── server/
    ├── benchmark/
    │   └── locust_runner.py       # Locust benchmark implementation
    ├── database/
    │   └── connection.py          # PostgreSQL connection handling
    ├── models/
    │   └── benchmark.py           # Data models
    ├── routes/
    │   ├── main_services.py       # Original API routes
    │   └── benchmark_routes.py    # Benchmark CRUD routes
    ├── schemas/
    │   └── benchmark.py           # Pydantic schemas
    ├── services/
    │   ├── benchmark_service.py   # Database operations
    │   └── benchmark_runner.py    # Benchmark execution
    └── utils/
        ├── security.py            # Authentication
        └── exceptions.py          # Custom exceptions
```

## API Endpoints

### Benchmark Operations
- `POST /api/v1/benchmarks/run` - Run new benchmark test
- `GET /api/v1/benchmarks/` - Get all benchmark results (paginated)
- `GET /api/v1/benchmarks/{id}` - Get specific benchmark result
- `PUT /api/v1/benchmarks/{id}` - Update benchmark result
- `DELETE /api/v1/benchmarks/{id}` - Delete benchmark result

### Original Endpoints
- `POST /api/v1/generate/` - Generate response (protected)
- `GET /api/v1/model/` - Get model information
- `GET /` - Health check

## Setup

1. **Clone and install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Setup PostgreSQL database**:
   ```bash
   # Create database
   createdb benchmark_db
   
   # Database will be initialized automatically on startup
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

## Environment Variables

### Database
- `DB_HOST` - PostgreSQL host (default: localhost)
- `DB_NAME` - Database name (default: benchmark_db)
- `DB_USER` - Database user (default: postgres)
- `DB_PASSWORD` - Database password
- `DB_PORT` - Database port (default: 5432)

### Authentication
- `BASIC_AUTH` - Enable HTTP Basic Auth (default: False)
- `API_USERNAME` - Basic auth username
- `API_PASSWORD` - Basic auth password
- `API_AUTH_TOKEN` - Bearer token for external APIs

### Benchmark Configuration
- `HUGGINGFACE_TOKEN` - HuggingFace token for model access
- `LOCUST_MODEL` - Default model name
- `LOCUST_TOKENIZER` - Default tokenizer name
- `LOCUST_HOST` - Default target API URL

## Usage Examples

### Run a Benchmark Test

```bash
curl -X POST "http://localhost:8000/api/v1/benchmarks/run" \
  -H "Authorization: Basic YWRtaW46c2VjdXJlX3Bhc3N3b3Jk" \
  -d "user=100&spawnrate=10&duration=60&url=https://api.example.com"
```

### Get All Results

```bash
curl -X GET "http://localhost:8000/api/v1/benchmarks/?page=1&limit=10" \
  -H "Authorization: Basic YWRtaW46c2VjdXJlX3Bhc3N3b3Jk"
```

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build
```

## Benchmark Metrics

The API collects and stores comprehensive metrics:

- **Time to First Token (TTFT)**: Latency to receive first response token
- **End-to-End Latency**: Total request completion time
- **Inter-Token Latency**: Average time between tokens
- **Token Speed**: Tokens generated per second per user
- **Throughput**: Input/output tokens per second across all users

## Development

The codebase follows clean architecture principles:

- **Routes**: FastAPI endpoint definitions
- **Services**: Business logic and orchestration
- **Models**: Data structures and database mapping
- **Schemas**: API request/response validation
- **Utils**: Common utilities and helpers

## Error Handling

Comprehensive error handling with custom exceptions:
- `BenchmarkError`: Benchmark operation failures
- `DatabaseError`: Database operation issues
- `ValidationError`: Input validation failures
- `URLValidationError`: Target URL accessibility issues