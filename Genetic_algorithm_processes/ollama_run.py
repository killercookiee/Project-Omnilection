"""
Genetic_algorithm_processes/ollama_run.py
"""

import json
import subprocess
import time
import re
from typing import List, Tuple, Dict, Any, Optional


class PromptChainRunner:
    """
    A class to manage and execute prompt chains using Ollama models.
    """
    
    def __init__(self,
                 model_registry_path: str = 'LLM_models/model_registry.json',
                 timeout: int = 120,
                 verbose: bool = False
    ):
        """
        Initialize the PromptChainRunner.
        
        Args:
            model_registry_path: Path to the model registry JSON file
        """
        self.model_registry_path = model_registry_path
        self.timeout = timeout
        self.verbose = verbose
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self.load_model_registry()
    
    def load_model_registry(self) -> None:
        try:
            with open(self.model_registry_path, 'r') as f:
                self.model_registry = json.load(f)
            if self.verbose:
                print(f"[PromptChainRunner] ✅ Registry loaded — {len(self.model_registry)} models from {self.model_registry_path}")
        except FileNotFoundError:
            print(f"[PromptChainRunner] ❌ Registry not found at {self.model_registry_path} — run update_model_registry() to create it")
        except json.JSONDecodeError as e:
            print(f"[PromptChainRunner] ❌ Error parsing registry: {e}")
    
    def parse_parameter_size(self, param_str: str) -> Optional[float]:
        """
        Parse parameter size string and convert to actual number.
        Handles formats like '620M', '1B', '494.03M', '1.5B', etc.
        
        Args:
            param_str: Parameter size string from ollama output
            
        Returns:
            Number of parameters, or None if parsing fails
        """
        if not param_str or param_str == "Unknown":
            return None
        
        try:
            # Remove any whitespace
            param_str = param_str.strip().upper()
            
            # Extract number and unit
            if param_str.endswith('B'):
                # Billion
                number = float(param_str[:-1])
                return number * 1_000_000_000
            elif param_str.endswith('M'):
                # Million
                number = float(param_str[:-1])
                return number * 1_000_000
            elif param_str.endswith('K'):
                # Thousand
                number = float(param_str[:-1])
                return number * 1_000
            else:
                # Try to parse as plain number
                return float(param_str)
        except (ValueError, AttributeError):
            return None
    
    def get_model_details(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model using 'ollama show'.
        
        Args:
            model_name: Name of the model to query
            
        Returns:
            Dictionary containing model details
        """
        details = {
            "architecture": "Unknown",
            "parameters": "Unknown",
            "parameters_count": None,  # Numeric value of parameters
            "context_length": 2048,
            "embedding_length": "Unknown",
            "quantization": "Unknown",
            "capabilities": [],
            "stop_tokens": [],
            "system_prompt": None,
            "model_parameters": {},
            "license": "Unknown"
        }
        
        try:
            result = subprocess.run(
                ['ollama', 'show', model_name],
                capture_output=True,
                text=True,
                check=True
            )
            
            output = result.stdout
            lines = output.split('\n')
            
            current_section = None
            system_lines = []
            license_lines = []
            
            for line in lines:
                stripped = line.strip()
                
                # Detect section headers
                if stripped.startswith('Model'):
                    current_section = 'model'
                    continue
                elif stripped.startswith('Capabilities'):
                    current_section = 'capabilities'
                    continue
                elif stripped.startswith('Parameters'):
                    current_section = 'parameters'
                    continue
                elif stripped.startswith('System'):
                    current_section = 'system'
                    continue
                elif stripped.startswith('License'):
                    current_section = 'license'
                    continue
                elif not stripped or stripped == '...':
                    continue
                
                # Parse based on current section
                if current_section == 'model':
                    if 'architecture' in stripped:
                        parts = stripped.split()
                        if len(parts) >= 2:
                            details['architecture'] = parts[-1]
                    elif 'parameters' in stripped:
                        parts = stripped.split()
                        if len(parts) >= 2:
                            param_str = parts[-1]
                            details['parameters'] = param_str
                            # Also parse to numeric value
                            details['parameters_count'] = self.parse_parameter_size(param_str)
                    elif 'context length' in stripped:
                        parts = stripped.split()
                        if len(parts) >= 2:
                            try:
                                details['context_length'] = int(parts[-1])
                            except ValueError:
                                pass
                    elif 'embedding length' in stripped:
                        parts = stripped.split()
                        if len(parts) >= 2:
                            details['embedding_length'] = parts[-1]
                    elif 'quantization' in stripped:
                        parts = stripped.split()
                        if len(parts) >= 2:
                            details['quantization'] = parts[-1]
                
                elif current_section == 'capabilities':
                    # Capabilities are single words on their own lines
                    if stripped:
                        details['capabilities'].append(stripped)
                
                elif current_section == 'parameters':
                    # Parse parameter key-value pairs
                    if 'stop' in stripped:
                        # Extract the stop token (usually in quotes)
                        match = re.search(r'"([^"]+)"', stripped)
                        if match:
                            details['stop_tokens'].append(match.group(1))
                    else:
                        # Other parameters like temperature, top_p, etc.
                        parts = stripped.split(None, 1)  # Split on first whitespace
                        if len(parts) == 2:
                            param_name = parts[0]
                            param_value = parts[1].strip()
                            # Try to convert to number if possible
                            try:
                                if '.' in param_value:
                                    param_value = float(param_value)
                                else:
                                    param_value = int(param_value)
                            except ValueError:
                                pass  # Keep as string
                            details['model_parameters'][param_name] = param_value
                
                elif current_section == 'system':
                    # System prompt can be multi-line
                    if stripped:
                        system_lines.append(stripped)
                
                elif current_section == 'license':
                    # Collect first few lines of license
                    if stripped and len(license_lines) < 2:
                        license_lines.append(stripped)
            
            # Combine system prompt lines
            if system_lines:
                details['system_prompt'] = ' '.join(system_lines)
            
            # Combine license lines
            if license_lines:
                details['license'] = ' '.join(license_lines)
            
        except subprocess.CalledProcessError as e:
            print(f"Error getting details for {model_name}: {e}")
        except Exception as e:
            print(f"Unexpected error getting details for {model_name}: {e}")
        
        return details
    
    def parse_duration(self, duration_str: str) -> float:
        """
        Parse duration string and convert to seconds.
        Handles formats like '2.5s', '500ms', '1.234567s', '229.277125ms',
        '5m0.708975167s', '4m27.737505291s', etc.
        """
        duration_str = duration_str.strip()

        # Handle combined minutes+seconds format: e.g. '5m0.708975167s' or '4m27.737505291s'
        match = re.match(r'([\d.]+)m([\d.]+)s', duration_str)
        if match:
            minutes = float(match.group(1))
            seconds = float(match.group(2))
            return minutes * 60 + seconds

        # Handle simple unit formats: '2.5s', '500ms', '229.277125ms', '5m', etc.
        match = re.search(r'([\d.]+)\s*([a-zµ]+)', duration_str)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            if unit == 's':
                return value
            elif unit == 'ms':
                return value / 1_000
            elif unit in ['µs', 'us']:
                return value / 1_000_000
            elif unit == 'ns':
                return value / 1_000_000_000
            elif unit == 'm':
                return value * 60

        return 0.0
    
    def parse_benchmark_output(self, stderr_output: str) -> Dict[str, Any]:
        """
        Parse the --verbose output from ollama run to extract timing and token information.

        Ollama --verbose prints stats to stderr in this format:
            total duration:       6.007939084s
            load duration:        1.555126s
            prompt eval count:    14 token(s)
            prompt eval duration: 229.277125ms
            prompt eval rate:     61.06 tokens/s
            eval count:           249 token(s)
            eval duration:        4.11188771s
            eval rate:            60.56 tokens/s

        All duration values are converted to seconds (float).
        Rate values are tokens/s (float).
        Token counts are integers.
        
        Args:
            stderr_output: The stderr output from ollama run --verbose
            
        Returns:
            Dictionary containing parsed benchmark metrics:
                - total_duration (float): Total wall-clock time in seconds
                - load_duration (float): Model load/warmup time in seconds
                - prompt_eval_count (int): Number of tokens in the prompt
                - prompt_eval_duration (float): Time to evaluate prompt in seconds
                - prompt_eval_rate (float): Prompt tokens processed per second
                - eval_count (int): Number of tokens generated
                - eval_duration (float): Time spent generating tokens in seconds
                - eval_rate (float): Generated tokens per second
        """
        metrics: Dict[str, Any] = {
            "total_duration": 0.0,
            "load_duration": 0.0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0.0,
            "prompt_eval_rate": 0.0,
            "eval_count": 0,
            "eval_duration": 0.0,
            "eval_rate": 0.0
        }
        
        for line in stderr_output.split('\n'):
            line = line.strip()
            
            if 'total duration:' in line:
                metrics['total_duration'] = self.parse_duration(line.split('total duration:')[1].strip())
            
            elif 'load duration:' in line:
                metrics['load_duration'] = self.parse_duration(line.split('load duration:')[1].strip())
            
            elif 'prompt eval count:' in line:
                match = re.search(r'(\d+)', line)
                if match:
                    metrics['prompt_eval_count'] = int(match.group(1))
            
            elif 'prompt eval duration:' in line:
                metrics['prompt_eval_duration'] = self.parse_duration(line.split('prompt eval duration:')[1].strip())
            
            elif 'prompt eval rate:' in line:
                match = re.search(r'([\d.]+)', line)
                if match:
                    metrics['prompt_eval_rate'] = float(match.group(1))
            
            # 'eval count' must be checked AFTER 'prompt eval count' to avoid partial match
            elif re.match(r'\s*eval count:', line):
                match = re.search(r'(\d+)', line)
                if match:
                    metrics['eval_count'] = int(match.group(1))
            
            elif re.match(r'\s*eval duration:', line):
                metrics['eval_duration'] = self.parse_duration(line.split('eval duration:')[1].strip())
            
            elif re.match(r'\s*eval rate:', line):
                match = re.search(r'([\d.]+)', line)
                if match:
                    metrics['eval_rate'] = float(match.group(1))
        
        return metrics
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory using the --keepalive 0 CLI hack.
        This ensures the model is removed from VRAM/RAM.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Run the model with --keepalive 0 to immediately unload it
            subprocess.run(
                ['ollama', 'run', model_name, '--keepalive', '0', 'unload'],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            # Give it a moment to fully unload
            time.sleep(0.5)
            return True
        except subprocess.TimeoutExpired:
            # Timeout is acceptable, model should be unloaded
            return True
        except Exception as e:
            print(f"Warning: Could not unload model {model_name}: {e}")
            return False
    
    def benchmark_model(self,
            model_name: str,
            test_prompt: str = "Write a haiku",
    ) -> Dict[str, Any]:
        """
        Benchmark a model with a 4-step process to capture different load states:
        1. Pre-warm: Load model from disk to RAM (not benchmarked, just warming)
        2. Unload: Remove model from VRAM/RAM using --keepalive 0
        3. Frozen Run (Cold): Load from RAM to VRAM (captures cold_load_duration)
        4. Warm Run: Model resident in VRAM (captures final benchmark metrics)
        
        Args:
            model_name: Name of the model to benchmark
            test_prompt: Prompt to use for benchmarking
            
        Returns:
            Dictionary containing benchmark results including cold/warm load times
        """
        benchmark_results = {
            "cold_load_duration": 0.0,
            "warm_load_duration": 0.0,
            "total_duration": 0.0,
            "load_duration": 0.0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0.0,
            "prompt_eval_rate": 0.0,
            "eval_count": 0,
            "eval_duration": 0.0,
            "eval_rate": 0.0
        }
        
        try:
            # Step 1: Pre-warm - Load model from disk to RAM (not benchmarked)
            print(f"  Pre-warming {model_name} (disk -> RAM)...")
            subprocess.run(
                ['ollama', 'run', model_name, test_prompt],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout
            )
            
            # Step 2: Unload - Remove from VRAM/RAM
            print(f"  Unloading {model_name} from VRAM...")
            self.unload_model(model_name)
            
            # Step 3: Frozen Run (Cold Load) - RAM to VRAM
            print(f"  Cold load test (RAM -> VRAM)...")
            result = subprocess.run(
                ['ollama', 'run', model_name, test_prompt, '--verbose'],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout
            )
            cold_metrics = self.parse_benchmark_output(result.stderr)
            benchmark_results['cold_load_duration'] = cold_metrics['load_duration']
            
            # Step 4: Warm Run - Model resident in VRAM
            print(f"  Warm load test (VRAM resident)...")
            result = subprocess.run(
                ['ollama', 'run', model_name, test_prompt, '--verbose'],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout
            )
            
            # Parse the warm run output for final benchmark
            final_metrics = self.parse_benchmark_output(result.stderr)
            
            # Update benchmark results with warm run data
            benchmark_results.update(final_metrics)
            # Preserve the cold load duration
            benchmark_results['cold_load_duration'] = cold_metrics['load_duration']
            # Store warm load duration separately
            benchmark_results['warm_load_duration'] = final_metrics['load_duration']
            
        except subprocess.CalledProcessError as e:
            print(f"Error benchmarking {model_name}: {e}")
        except Exception as e:
            print(f"Unexpected error benchmarking {model_name}: {e}")
        
        return benchmark_results
    
    def update_model_registry(self, benchmark: bool = True, benchmark_model: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Update the model registry by fetching detailed information from Ollama.

        Merges into the existing registry file rather than replacing it, so
        previously recorded benchmark data is preserved for models that are
        not re-benchmarked in this run (i.e. when benchmark=False).
        When benchmark=True, only the 'benchmark' key for each model is
        overwritten — all other existing keys are preserved as well.

        Models that have been deleted from Ollama are left in the registry
        untouched (they simply won't appear in 'ollama list').
        
        Args:
            benchmark: Whether to run benchmarks on each model (can be slow).
                       If False, any existing 'benchmark' entries in the file
                       are preserved as-is.
        
        Returns:
            Dictionary containing the full (merged) model registry.
        """
        import os

        try:
            # Run 'ollama list' command
            result = subprocess.run(
                ['ollama', 'list'], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # Parse the output
            lines = result.stdout.strip().split('\n')
            
            # Skip the header line
            model_lines = lines[1:] if len(lines) > 1 else []
            
            # Load existing registry from disk so we can merge into it.
            # self.model_registry may already be populated from __init__, but
            # re-reading here ensures we pick up any changes made on disk since
            # the object was created (e.g. by another process).
            existing_registry: Dict[str, Dict[str, Any]] = {}
            if os.path.exists(self.model_registry_path):
                try:
                    with open(self.model_registry_path, 'r') as f:
                        existing_registry = json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    print(f"Warning: could not read existing registry ({e}). Starting fresh.")

            for line in model_lines:
                # Parse each line (format: NAME    ID    SIZE    MODIFIED)
                parts = line.split()
                if not parts:
                    continue

                model_name = parts[0]
                model_id   = parts[1] if len(parts) > 1 else "Unknown"
                size_str   = parts[2] if len(parts) > 2 else "Unknown"
                
                print(f"\nProcessing model: {model_name}")
                
                # Get detailed model information
                print(f"  Fetching details...")
                details = self.get_model_details(model_name)
                
                # Build the freshly-fetched fields (everything except benchmark)
                fresh_fields = {
                    "model_id": model_id,
                    "size": size_str,
                    "cost_per_1k_tokens": 0.0,  # Ollama models are free and local
                    "architecture": details['architecture'],
                    "parameters": details['parameters'],
                    "parameters_count": details['parameters_count'],
                    "context_length": details['context_length'],
                    "embedding_length": details['embedding_length'],
                    "quantization": details['quantization'],
                    "capabilities": details['capabilities'],
                    "stop_tokens": details['stop_tokens'],
                    "system_prompt": details['system_prompt'],
                    "model_parameters": details['model_parameters'],
                    "license": details['license'],
                }

                # Start from the existing entry (preserves benchmark + any
                # custom fields added manually), then overlay fresh fields.
                merged = dict(existing_registry.get(model_name, {}))
                merged.update(fresh_fields)

                # Optionally overwrite the benchmark key
                if benchmark and (benchmark_model is None or benchmark_model == model_name):
                    print(f"  Running benchmark...")
                    merged["benchmark"] = self.benchmark_model(model_name)
                elif "benchmark" in merged:
                    print(f"  Skipping benchmark — keeping existing data.")
                
                existing_registry[model_name] = merged
                print(f"  ✓ Completed {model_name}")
            
            # Persist the merged registry
            self.model_registry = existing_registry
            os.makedirs(os.path.dirname(self.model_registry_path), exist_ok=True)
            
            with open(self.model_registry_path, 'w') as f:
                json.dump(self.model_registry, f, indent=4)
            
            print(f"\n✓ Model registry updated successfully with {len(self.model_registry)} models")
            print(f"✓ Saved to {self.model_registry_path}")
            return self.model_registry
            
        except subprocess.CalledProcessError as e:
            print(f"Error running 'ollama list': {e}")
            print("Make sure Ollama is installed and running")
            return {}
        except Exception as e:
            print(f"Error updating model registry: {e}")
            return {}
    
    def run_ollama_model(self, model_name: str, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Run a single Ollama model with the given prompt.

        Uses --verbose to capture Ollama's runtime stats from stderr, which are
        parsed into a structured metrics dict. All durations are in seconds.
        
        Args:
            model_name: Name of the Ollama model
            prompt: The prompt to send to the model
        Returns:
            Tuple of (output_text, runtime_metrics) where runtime_metrics is a dict:
                - total_duration (float): Total wall-clock time in seconds
                - load_duration (float): Model load/warmup time in seconds
                - prompt_eval_count (int): Number of tokens in the prompt
                - prompt_eval_duration (float): Time to evaluate the prompt in seconds
                - prompt_eval_rate (float): Prompt tokens processed per second
                - eval_count (int): Number of tokens generated
                - eval_duration (float): Time spent generating tokens in seconds
                - eval_rate (float): Generated tokens per second
        """
        empty_metrics: Dict[str, Any] = {
            "total_duration": 0.0,
            "load_duration": 0.0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0.0,
            "prompt_eval_rate": 0.0,
            "eval_count": 0,
            "eval_duration": 0.0,
            "eval_rate": 0.0,
        }

        try:
            result = subprocess.run(
                ['ollama', 'run', model_name, prompt, '--verbose'],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout
            )
            
            output = result.stdout.strip()
            metrics = self.parse_benchmark_output(result.stderr)

            if self.verbose:
                status = "✅" if metrics['total_duration'] > 0 else "⚠️"
                print(f"\n{'─'*40}")
                print(f"  {status}  {model_name}")
                print(f"      Duration : {metrics['total_duration']:.3f}s  (load: {metrics['load_duration']:.3f}s)")
                print(f"      Tokens   : {metrics['prompt_eval_count']} prompt → {metrics['eval_count']} generated  ({metrics['eval_rate']:.1f} tok/s)")
                print(f"      Output   : {output[:80]}{'...' if len(output) > 80 else ''}")
                print(f"{'─'*40}")
            
            return (output, metrics)
            
        except subprocess.CalledProcessError as e:
            print(f"Error running model {model_name}: {e}")
            return (f"Error: {str(e)}", empty_metrics)
        except subprocess.TimeoutExpired as e:
            print(f"Timeout running model {model_name}: {e}")
            return (f"Timeout: {str(e)}", empty_metrics)
        except Exception as e:
            print(f"Unexpected error with model {model_name}: {e}")
            return (f"Error: {str(e)}", empty_metrics)
    
    def run_prompt_chain(
        self, 
        prompt_chain: List[Tuple[str, ...]], 
        initial_input: str,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Run a chain of prompts through different models sequentially.
        
        Args:
            prompt_chain: List of tuples, each containing (model_name, *prompt_segments)
            initial_input: The initial input to start the chain
            verbose: Whether to print progress information
            timeout: Maximum time in seconds to wait for each model run

        Returns:
            List of tuples containing (output, runtime_metrics) for each step.

            runtime_metrics keys (all per-step, not accumulated):
                - total_duration (float): Total wall-clock time in seconds
                - load_duration (float): Model load time in seconds
                - prompt_eval_count (int): Number of prompt tokens
                - prompt_eval_duration (float): Prompt evaluation time in seconds
                - prompt_eval_rate (float): Prompt tokens/s
                - eval_count (int): Number of generated tokens
                - eval_duration (float): Generation time in seconds
                - eval_rate (float): Generated tokens/s
        """
        # Evaluate prompt chain validity
        is_valid, missing_models = self.validate_prompt_chain(prompt_chain)
        if not is_valid:
            raise ValueError(f"Invalid prompt chain. Missing models: {', '.join(missing_models)}")
        
        results: List[Tuple[str, Dict[str, Any]]] = []
        current_context = initial_input
        if self.verbose:
            print(f"\n\n=== Running prompt chain with {len(prompt_chain)} steps... ===")
        
        for step_idx, step in enumerate(prompt_chain):
            # Extract model name and prompt segments
            model_name = step[0]
            prompt_segments = step[1:]
            
            # Combine prompt segments into a single prompt
            prompt_text = "".join(prompt_segments)
            
            # Add context from previous step
            if step_idx == 0:
                full_prompt = f"{prompt_text}\n\nInput: {current_context}"
            else:
                full_prompt = f"{prompt_text}\n\nPrevious output: {current_context}"
            
            if self.verbose:
                print(f"\n--- Step {step_idx + 1} ---")
                print(f"Model: {model_name}")
                print(f"Prompt preview: {full_prompt[:100]}...")
            
            # Run the model
            output, metrics = self.run_ollama_model(model_name, full_prompt)
            
            # Store result
            results.append((output, metrics))
            
            # Update context for next iteration
            current_context = output
            
            if self.verbose:
                print(f"Total duration:   {metrics['total_duration']:.3f}s")
                print(f"Load duration:    {metrics['load_duration']:.3f}s")
                print(f"Prompt tokens:    {metrics['prompt_eval_count']}  ({metrics['prompt_eval_rate']:.2f} tok/s)")
                print(f"Generated tokens: {metrics['eval_count']}  ({metrics['eval_rate']:.2f} tok/s)")
                print(f"Output preview:   {output[:100]}...")
        
        return results
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available models in the registry.
        
        Returns:
            List of model names
        """
        return list(self.model_registry.keys())
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information or None if not found
        """
        return self.model_registry.get(model_name)
    
    def validate_prompt_chain(self, prompt_chain: List[Tuple[str, ...]]) -> Tuple[bool, List[str]]:
        """
        Validate that all models in a prompt chain are available.
        
        Args:
            prompt_chain: List of tuples containing (model_name, *prompt_segments)
            
        Returns:
            Tuple of (is_valid, list_of_missing_models)
        """
        missing_models = []
        
        for step in prompt_chain:
            model_name = step[0]
            if model_name not in self.model_registry:
                missing_models.append(model_name)
        
        is_valid = len(missing_models) == 0
        return (is_valid, missing_models)
    
    def print_model_summary(self, model_name: str) -> None:
        """
        Print a formatted summary of a model's information.
        
        Args:
            model_name: Name of the model to summarize
        """
        info = self.get_model_info(model_name)
        if not info:
            print(f"Model '{model_name}' not found in registry")
            return
        
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")
        print(f"Architecture:       {info.get('architecture', 'N/A')}")
        
        # Show both string and numeric parameters
        param_str = info.get('parameters', 'N/A')
        param_count = info.get('parameters_count')
        if param_count:
            # Format with commas for readability
            param_formatted = f"{param_str} ({param_count:,} parameters)"
        else:
            param_formatted = param_str
        print(f"Parameters:         {param_formatted}")
        
        print(f"Context Length:     {info.get('context_length', 'N/A')}")
        print(f"Embedding Length:   {info.get('embedding_length', 'N/A')}")
        print(f"Quantization:       {info.get('quantization', 'N/A')}")
        print(f"Size:               {info.get('size', 'N/A')} MB")
        
        # Capabilities
        if info.get('capabilities'):
            caps = ', '.join(info['capabilities'])
            print(f"Capabilities:       {caps}")
        
        # Stop tokens
        if info.get('stop_tokens'):
            stops = ', '.join([f'"{s}"' for s in info['stop_tokens']])
            print(f"Stop Tokens:        {stops}")
        
        # Model parameters (temperature, top_p, etc.)
        if info.get('model_parameters'):
            print(f"\nModel Parameters:")
            for param, value in info['model_parameters'].items():
                print(f"  {param:18s}: {value}")
        
        # System prompt
        if info.get('system_prompt'):
            sys_prompt = info['system_prompt']
            # Truncate if too long
            if len(sys_prompt) > 100:
                sys_prompt = sys_prompt[:97] + "..."
            print(f"\nSystem Prompt:      {sys_prompt}")
        
        # License
        print(f"License:            {info.get('license', 'N/A')}")
        
        # Benchmark results
        if 'benchmark' in info:
            bench = info['benchmark']
            print(f"\nBenchmark Results:")
            print(f"  Cold Load (RAM→VRAM):  {bench.get('cold_load_duration', 0):.3f}s")
            print(f"  Warm Load (cached):    {bench.get('warm_load_duration', 0):.3f}s")
            print(f"  Total Duration:        {bench.get('total_duration', 0):.3f}s")
            print(f"  Prompt Eval Rate:      {bench.get('prompt_eval_rate', 0):.2f} tokens/s")
            print(f"  Generation Rate:       {bench.get('eval_rate', 0):.2f} tokens/s")
            print(f"  Prompt Tokens:         {bench.get('prompt_eval_count', 0)}")
            print(f"  Generated Tokens:      {bench.get('eval_count', 0)}")
        
        print(f"{'='*70}\n")


#%% Example usage:
if __name__ == "__main__":
    # Initialize the runner
    runner = PromptChainRunner(verbose=True, timeout=120)
    

    # ============ MODEL REGISTRY AND BENCHMARKING ============
    # Update the model registry with benchmarking
    print("Updating model registry from Ollama...")
    print("This may take a while as each model is benchmarked...\n")
    runner.update_model_registry(benchmark=True, benchmark_model="codellama:latest")  # Set to None to benchmark all models, or specify a single model name to benchmark just that one.
    
    print(f"\nAvailable models: {runner.get_available_models()}")
    
    # Print detailed info for each model
    for model_name in runner.get_available_models():
        runner.print_model_summary(model_name)
    

    # ============ RUNNING A PROMPT CHAIN ============
    # Example prompt chain (update model names to match your Ollama models)
    prompt_chain = [
        ("smollm:135m", "Summarize this: ", "Be concise."), 
        ("qwen:0.5b", "Improve ", "the following text: ")
    ]
    
    initial_input = "What is the capital of France?"
    
    # Validate the prompt chain
    is_valid, missing = runner.validate_prompt_chain(prompt_chain)
    if not is_valid:
        print(f"\nWarning: Missing models: {missing}")
        exit(1)
    else:
        print("\nPrompt chain is valid. Running the chain...")

    # Run the prompt chain
    print("\n\n========== RUNNING PROMPT CHAIN ==========")
    results = runner.run_prompt_chain(prompt_chain, initial_input)
    # results is a list of (output, metrics_dict) tuples for each step in the chain
    
    print("\n\n========== RESULTS ==========")
    for idx, (output, metrics) in enumerate(results):
        print(f"\nStep {idx + 1}:")
        print(f"  Output:           {output}")
        print(f"  Total duration:   {metrics['total_duration']:.3f}s")
        print(f"  Load duration:    {metrics['load_duration']:.3f}s")
        print(f"  Prompt tokens:    {metrics['prompt_eval_count']}  ({metrics['prompt_eval_rate']:.2f} tok/s)")
        print(f"  Generated tokens: {metrics['eval_count']}  ({metrics['eval_rate']:.2f} tok/s)")