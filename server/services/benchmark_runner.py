import gevent.monkey
gevent.monkey.patch_all()

import os
import subprocess
import logging
import requests
from openai import OpenAI
from typing import Dict, Any, Optional
from server.benchmark.locust_runner import get_current_metrics
from dotenv import load_dotenv
from transformers import AutoTokenizer 
from huggingface_hub import list_repo_files

load_dotenv()
logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Service class to run benchmark tests"""
    
    @staticmethod
    def validate_url(url: str, api_key: str) -> bool:
        """Validate if the target URL is accessible"""
        try:
            logger.info(f"Testing url = {url}")
            response = requests.get(
                f"{url}/v1/models", 
                headers={"Authorization": f"Bearer {api_key}"}, 
                timeout=10
            )
            logger.info(f"URL validation response status: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"URL validation failed: {e}")
            return False
    
    @staticmethod
    def get_default_model(url: str, api_key: str) -> str:
        """Get default model from the target API"""
        try:
            model_endpoint = f"{url}/v1/models"
            response = requests.get(
                model_endpoint, 
                headers={"Authorization": f"Bearer {api_key}"}
            )
            
            if response.status_code == 200:
                model_data = response.json()
                logger.info(f"Available models: {model_data}")
                return model_data.get("data", [{}])[0].get("id", "meta-llama/Llama-3.2-90B-Vision-Instruct")
            else:
                logger.warning(f"Failed to fetch models, status: {response.status_code}")
                return "meta-llama/Llama-3.2-90B-Vision-Instruct"
                
        except Exception as e:
            logger.error(f"Error fetching default model: {e}")
            return "meta-llama/Llama-3.2-90B-Vision-Instruct"
    
    @staticmethod
    def run_benchmark(
        users: int,
        spawn_rate: int,
        duration: int,
        url: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        tokenizer: Optional[str] = None,
        dataset: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the benchmark test and return results"""
        
        try:
            # Validate URL first
            if not BenchmarkRunner.validate_url(url, api_key):
                raise Exception(f"Target URL {url} is not accessible")
            
            # Get model if not provided
            if model is None:
                logger.info("Model not provided, fetching default model")
                model = BenchmarkRunner.get_default_model(url, api_key)
            
            # Set tokenizer to model if not provided
            if tokenizer is None:
                try:
                    repo_files = list_repo_files(
                        repo_id=model,
                        token=os.getenv('HUGGINGFACE_TOKEN')
                    )
                    if "tokenizer.json" in repo_files or "tokenizer_config.json" in repo_files:
                        tokenizer=model
                        logger.info(f"set {tokenizer} as a token from model")
                except:
                    client = OpenAI(
                        base_url=os.getenv('LOCUST_HOST'),
                        api_key=api_key
                    )
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user"}],
                        temperature=0.5,
                        max_tokens=10
                    )
                    token=response.model
                    repo_files = list_repo_files(
                        repo_id=token,
                        token=os.getenv('HUGGINGFACE_TOKEN')
                    )
                    if "tokenizer.json" in repo_files or "tokenizer_config.json" in repo_files:
                        tokenizer=token      
                        logger.info(f"set {tokenizer} as a token from huggingface")
            
            # Set dataset default if not provided
            if dataset is None:
                logger.info("Dataset not provided, using default")
                dataset = "mteb/banking77"
            
            logger.info(f"Using model: {model}")
            logger.info(f"Using tokenizer: {tokenizer}")
            logger.info(f"Using dataset: {dataset}")
            
            # Clear any existing metrics file to ensure fresh results
            import tempfile
            metrics_file = os.path.join(tempfile.gettempdir(), 'locust_metrics.json')
            if os.path.exists(metrics_file):
                os.remove(metrics_file)
                logger.info("Cleared existing metrics file")
            
            # Set environment variables for Locust
            env_vars = {
                "LOCUST_USERS": str(users),
                "LOCUST_SPAWN_RATE": str(spawn_rate),
                "LOCUST_DURATION": str(duration),
                "LOCUST_HOST": str(url),
                "LOCUST_MODEL": model,
                "LOCUST_API_KEY": api_key,
                "LOCUST_TOKENIZER": tokenizer,
                "LOCUST_DATASET": dataset
            }
            
            # Update environment
            for key, value in env_vars.items():
                os.environ[key] = value
            
            # Build Locust command
            locust_command = [
                "locust",
                "-f", "server/benchmark/locust_runner.py",
                "--headless",
                "--users", str(users),
                "--spawn-rate", str(spawn_rate),
                "--run-time", f"{duration}s",
                "--host", url
            ]
            
            logger.info("Starting Locust benchmark")
            logger.info(f"Command: {' '.join(locust_command)}")
            
            # Start the Locust process and capture output
            process = subprocess.Popen(
                locust_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Wait for the process to complete
            stdout, stderr = process.communicate()
            
            if stderr:
                logger.warning(f"Locust process stderr: {stderr}")
            
            if process.returncode != 0:
                raise Exception(f"Locust process failed with return code {process.returncode}")
            
            # Small delay to ensure metrics file is fully written
            import time
            time.sleep(1)
            
            # Get metrics
            metrics = get_current_metrics()
            logger.info(f"Benchmark completed, metrics: {metrics}")
            
            return {
                "status": "Test completed",
                "metrics": metrics.get("metrics", {}),
                "configuration": {
                    "user": users,
                    "spawnrate": spawn_rate,
                    "model": model,
                    "tokenizer": tokenizer,
                    "url": url,
                    "duration": duration,
                    "dataset": dataset
                }
            }
            
        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            raise Exception(f"Benchmark failed: {str(e)}")