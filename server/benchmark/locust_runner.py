#TODO: Make this script more elegant and robust.

import os
import time
import random
import json
import uuid
import statistics
import threading
from locust import HttpUser, task, between, events
from transformers import AutoTokenizer 
from datasets import load_dataset 
import tempfile
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load default environment variables
load_dotenv()

# Global variables that will be set when needed
tokenizer = None
prompts = None
model_name = None
_initialization_lock = threading.Lock()
_initialized = False

def initialize_benchmark_resources():
    try:
        """Initialize tokenizer and dataset when actually running a benchmark (thread-safe)"""
        global tokenizer, prompts, model_name, _initialized
        
        # Thread-safe check
        if _initialized:
            return
        
        with _initialization_lock:
            # Double-check after acquiring lock
            if _initialized:
                return
                
            model_name = str(os.getenv("LOCUST_MODEL"))
            tokenizer_name = str(os.getenv("LOCUST_TOKENIZER"))
            dataset_name = str(os.getenv("LOCUST_DATASET", "mteb/banking77"))
            
            if not model_name or model_name == "None":
                raise ValueError("LOCUST_MODEL environment variable is required")
            if not tokenizer_name or tokenizer_name == "None":
                raise ValueError("LOCUST_TOKENIZER environment variable is required")
            
            logger.info(f"Starting tokenizer download for: {tokenizer_name}")
            # Initialize tokenizer with user-specified tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
            logger.info(f"Tokenizer downloaded successfully: {tokenizer_name}")

            logger.info(f"Starting dataset download for: {dataset_name}")
            # Load dataset prompts
            dataset = load_dataset(dataset_name, split="test")
            prompts = dataset["text"]
            logger.info(f"Dataset downloaded successfully: {dataset_name} ({len(prompts)} prompts)")
            
            _initialized = True
    except Exception as e:
        logger.error(f"Error in benchmark init: {str(e)}")
        raise

# Global variables for metrics collection
ttft_times = []
end_to_end_latencies = []
inter_token_latencies = []
tokens_per_second_list = []
start_benchmark_time = None
total_input_tokens = 0
total_output_tokens = 0

# Create a temporary file path for metrics
METRICS_FILE = os.path.join(tempfile.gettempdir(), 'locust_metrics.json')

def calculate_stats(data):
    if not data:
        return {
            "average": 0,
            "maximum": 0,
            "minimum": 0,
            "median": 0,
        }
    return {
        "average": round(sum(data) / len(data), 2),
        "maximum": round(max(data), 2),
        "minimum": round(min(data), 2),
        "median": round(statistics.median(data), 2),
    }

class LLMBenchmarkUser(HttpUser):
    # Set wait time between tasks to 0.5 to 5 seconds
    wait_time = between(0.5, 5)
    logger.info("initiate response")

    @task(1)
    def generate_response(self):
        global total_input_tokens, total_output_tokens, start_benchmark_time
        global ttft_times, end_to_end_latencies, inter_token_latencies, tokens_per_second_list
        logger.info("initiate response")
        # Initialize resources on first call
        initialize_benchmark_resources()
        
        if start_benchmark_time is None:
            start_benchmark_time = time.time()

        # Track request start time
        start_time = time.time()
        first_token_time = None
        tokens = []
        
        try:
            # Select a random prompt and append UUID
            input_text = f"{random.choice(prompts)} {uuid.uuid4()}"
            input_length = len(tokenizer(input_text)['input_ids'])
            total_input_tokens += input_length

            logger.info(f"Making HTTP request to /v1/chat/completions with model: {model_name}")

            # Send request
            headers = {
                'Content-type': 'application/json',
                'Accept': 'application/json'
            }
            
            # Only add auth header if token is provided and not a placeholder
            auth_token = os.getenv("API_AUTH_TOKEN")
            if auth_token and auth_token != "your_api_token_here":
                headers['Authorization'] = f'Bearer {auth_token}'

            response = self.client.post(
                url="/v1/chat/completions",
                headers=headers,
                data=json.dumps({
                    "model": model_name,
                    "messages": [{"role": "user", "content": input_text}],
                    "stream": True,
                    "temperature": 0.9,
                    "top_p": 0.9,
                    "max_tokens": 128,
                    "min_tokens": 20
                }),
                stream=True
            )
            
            logger.info(f"HTTP response status: {response.status_code}")

            # Process streamed response
            for line in response.iter_lines():
                if line:
                    token_time = time.time()
                    if first_token_time is None:
                        first_token_time = token_time
                        ttft = (first_token_time - start_time) * 1000
                        ttft_times.append(ttft)
                    tokens.append(line)

            # Calculate metrics
            end_time = time.time()
            e2e_latency = (end_time - start_time) * 1000
            end_to_end_latencies.append(e2e_latency)

            output_length = len(tokens)
            total_output_tokens += output_length

            if len(tokens) > 1:
                itl = ((end_time - first_token_time) / (len(tokens) - 1)) * 1000
                inter_token_latencies.append(itl)

            token_speed = output_length / (end_time - start_time)
            tokens_per_second_list.append(token_speed)

            # Debug print
            logger.debug(f"Request completed - TTFT: {ttft_times[-1]:.2f}ms, E2E: {e2e_latency:.2f}ms")

        except Exception as e:
            logger.error(f"Error in benchmark request: {str(e)}")
            raise

@events.quitting.add_listener
def display_metrics_summary(environment, **kwargs):
    """Calculate and save metrics when benchmark completes"""
    global total_input_tokens, total_output_tokens
    
    logger.info(f"Collected data points:")
    logger.info(f"TTFT times: {len(ttft_times)} points")
    logger.info(f"E2E latencies: {len(end_to_end_latencies)} points")
    logger.info(f"Inter-token latencies: {len(inter_token_latencies)} points")
    logger.info(f"Token speeds: {len(tokens_per_second_list)} points")

    # Calculate stats only if we have data
    if ttft_times:
        # Calculate benchmark duration
        benchmark_duration = time.time() - start_benchmark_time if start_benchmark_time else 0

        # Calculate throughput
        input_token_throughput = total_input_tokens / benchmark_duration if benchmark_duration > 0 else 0
        output_token_throughput = total_output_tokens / benchmark_duration if benchmark_duration > 0 else 0

        # Calculate stats
        ttft_stats = calculate_stats(ttft_times)
        e2e_stats = calculate_stats(end_to_end_latencies)
        inter_token_stats = calculate_stats(inter_token_latencies)
        token_speed_stats = calculate_stats(tokens_per_second_list)

        # Store metrics in a temporary file
        metrics = {
            "metrics": {
                "time_to_first_token": ttft_stats,
                "end_to_end_latency": e2e_stats,
                "inter_token_latency": inter_token_stats,
                "token_speed": token_speed_stats,
                "throughput": {
                    "input_tokens_per_second": round(input_token_throughput, 2),
                    "output_tokens_per_second": round(output_token_throughput, 2)
                }
            }
        }
        
        # Save metrics to temporary file
        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics, f)
        
        logger.info("Metrics saved to temporary file")

def get_current_metrics():
    """Get current metrics from temporary file"""
    try:
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "metrics": {
                "time_to_first_token": calculate_stats(ttft_times),
                "end_to_end_latency": calculate_stats(end_to_end_latencies),
                "inter_token_latency": calculate_stats(inter_token_latencies),
                "token_speed": calculate_stats(tokens_per_second_list),
                "throughput": {
                    "input_tokens_per_second": 0,
                    "output_tokens_per_second": 0
                }
            }
        }

# Comment out LoadTestShape to use default Locust behavior with command-line args
# class StagesShape(LoadTestShape):
#     """
#     Fixed staged load pattern that runs through predefined stages sequentially
#     """

#     def tick(self):
#         run_time = self.get_run_time()
        
#         if run_time < locust_duration:
#             return (locust_users, locust_spawn_rate)
#         return None

# Initialize benchmark start time
start_benchmark_time = time.time()