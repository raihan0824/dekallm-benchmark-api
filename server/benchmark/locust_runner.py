import os
import time
import random
import json
import uuid
import statistics
from locust import HttpUser, task, between, events 
import tempfile
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load default environment variables
load_dotenv()

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
    """Calculate statistics for a list of values"""
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

def initialize_tokenizer_and_dataset():
    """Initialize tokenizer and dataset for benchmark"""
    try:
        model_name = str(os.getenv("LOCUST_MODEL", "deepseek-ai/DeepSeek-R1-70B"))
        tokenizer_name = str(os.getenv("LOCUST_TOKENIZER", "deepseek-ai/DeepSeek-R1"))
        dataset_name = str(os.getenv("LOCUST_DATASET", "mteb/banking77"))
        
        # Initialize tokenizer with user-specified tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        
        # Load dataset prompts (configurable)
        dataset = load_dataset(dataset_name, split="test")
        prompts = dataset["text"]
        
        logger.info(f"Initialized with dataset: {dataset_name}")
        return tokenizer, prompts
    except Exception as e:
        logger.error(f"Error initializing tokenizer and dataset: {e}")
        raise

class LLMBenchmarkUser(HttpUser):
    """Locust user class for LLM benchmarking"""
    wait_time = between(0.5, 5)
    
    def on_start(self):
        """Initialize tokenizer and prompts for this user"""
        self.tokenizer, self.prompts = initialize_tokenizer_and_dataset()

    @task()
    def generate_response(self):
        """Main benchmark task - send LLM generation request"""
        global total_input_tokens, total_output_tokens, start_benchmark_time
        global ttft_times, end_to_end_latencies, inter_token_latencies, tokens_per_second_list

        if start_benchmark_time is None:
            start_benchmark_time = time.time()

        # Track request start time
        start_time = time.time()
        first_token_time = None
        tokens = []
        
        try:
            # Get environment variables
            model_name = str(os.getenv("LOCUST_MODEL", "deepseek-ai/DeepSeek-R1-70B"))
            
            # Select a random prompt and append UUID
            input_text = f"{random.choice(self.prompts)} {uuid.uuid4()}"
            input_length = len(self.tokenizer(input_text)['input_ids'])
            total_input_tokens += input_length

            # Send request
            headers = {
                'Content-type': 'application/json',
                'Accept': 'application/json'
            }
            
            # # Only add auth header if token is provided and not a placeholder
            # auth_token = os.getenv("API_AUTH_TOKEN")
            # if auth_token and auth_token != "your_api_token_here":
            #     headers['Authorization'] = f'Bearer {auth_token}'
            
            response = self.client.post(
                url="/v1/chat/completions",
                headers=headers,
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": input_text}],
                    "stream": True,
                    "temperature": 0.9,
                    "top_p": 0.9,
                    "max_tokens": 128,
                    "min_tokens": 20
                },
                stream=True
            )

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

        except Exception as e:
            logger.error(f"Error in benchmark request: {str(e)}")
            # Log additional debug info
            logger.error(f"Request URL: /v1/chat/completions")
            logger.error(f"Model: {model_name}")
            if 'response' in locals():
                logger.error(f"Response status: {response.status_code if hasattr(response, 'status_code') else 'Unknown'}")
                try:
                    logger.error(f"Response text: {response.text[:500] if hasattr(response, 'text') else 'No text'}")
                except:
                    logger.error("Could not get response text")
            raise  # Re-raise the exception so Locust marks the request as failed

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

# Commenting out LoadTestShape to use default Locust behavior with command-line args
# class StagesShape(LoadTestShape):
#     """Fixed staged load pattern that runs through predefined stages sequentially"""

#     def tick(self):
#         run_time = self.get_run_time()
        
#         locust_users = int(os.getenv("LOCUST_USERS", 100))
#         locust_spawn_rate = int(os.getenv("LOCUST_SPAWN_RATE", 100))
#         locust_duration = int(os.getenv("LOCUST_DURATION", 60))
        
#         if run_time < locust_duration:
#             return (locust_users, locust_spawn_rate)
#         return None

# Initialize benchmark start time
start_benchmark_time = time.time()