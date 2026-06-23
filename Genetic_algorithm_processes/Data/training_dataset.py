"""
Genetic_algorithm_processes/Data/training_dataset.py
"""

class TrainingDatasetManager:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def load_industry_benchmarks(self, num_math: int = 5, num_coding: int = 5) -> list[dict]:
        """
        Attempts to load industry-standard benchmarks from Hugging Face.
        Falls back to local multi-axis datasets if offline or if 'datasets' is not installed.
        """
        training_data = []
        
        try:
            from datasets import load_dataset
            
            # ── 1. Load GSM8K (Math / Chain-of-Thought) ──
            if self.verbose:
                print(f"[DatasetManager] Downloading GSM8K Benchmark ({num_math} samples)...")
            gsm8k = load_dataset("gsm8k", "main", split="train")
            
            for item in list(gsm8k)[:num_math]:
                expected_number = item['answer'].split('####')[-1].strip()
                training_data.append({
                    "task_type": "math",
                    "input": f"Solve this step-by-step, but your final sentence MUST be exactly: 'The final answer is {expected_number}.' Problem: {item['question']}",
                    "output": f"The final answer is {expected_number}."
                })

            # ── 2. Load MBPP (Python Coding Benchmark) ──
            if self.verbose:
                print(f"[DatasetManager] Downloading MBPP Benchmark ({num_coding} samples)...")
            mbpp = load_dataset("mbpp", "sanitized", split="test")
            
            for item in list(mbpp)[:num_coding]:
                training_data.append({
                    "task_type": "coding",
                    "input": f"Write a Python function to solve this: {item['prompt']}. Requirements: 1) Must pass this test case: {item['test_list'][0]}. 2) Output ONLY raw code.",
                    "output": item['code']
                })
                
        except ImportError:
            print("[DatasetManager] ⚠️ 'datasets' library not found. Falling back to local dataset.")
            return self._get_local_fallback()
        except Exception as e:
            print(f"[DatasetManager] ⚠️ Failed to download benchmarks ({e}). Falling back to local dataset.")
            return self._get_local_fallback()

        if self.verbose:
            print(f"[DatasetManager] ✅ Successfully loaded {len(training_data)} benchmark tasks.")
            
        return training_data

    def _get_local_fallback(self) -> list[dict]:
        """Provides a local multi-axis dataset if Hugging Face fails."""
        return [
            {
                "task_type": "coding_architecture",
                "input": "Write a Python class for an LRU (Least Recently Used) Cache. Requirements: 1) Initialize with a capacity. 2) get(key) and put(key, value) methods. 3) Explain your reasoning before writing the code.",
                "output": (
                    "Assess the AI's response across these 4 criteria:\n"
                    "1. Syntax: 0 errors, valid Python code.\n"
                    "2. Architecture: Uses optimal data structures (e.g., OrderedDict or a Dict + Doubly Linked List).\n"
                    "3. Logic: get() and put() correctly update recency and genuinely operate in O(1) time.\n"
                    "4. Understandability: Includes clear comments and clean naming conventions.\n"
                    "Score 1.0 if all 4 are perfect. Deduct 0.25 for each criteria that is missing, flawed, or non-optimal. Output ONLY the float."
                )
            },
            {
                "task_type": "debugging",
                "input": "A developer wrote this Python code to remove duplicates while iterating:\n`for item in my_list:\n  if my_list.count(item) > 1:\n    my_list.remove(item)`\nIdentify the bug, explain why it happens, and provide a robust fix.",
                "output": (
                    "Assess the AI's response across these 3 criteria:\n"
                    "1. Error Tracing: Clearly identifies that modifying a list while iterating over it causes index shifting and skipped elements.\n"
                    "2. Solution Complexity: Provides a systemic, pythonic fix (e.g., `list(set(my_list))` or `dict.fromkeys(my_list)`) rather than a hacky while-loop.\n"
                    "3. Testability/Integrity: The solution safely preserves data integrity and handles edge cases.\n"
                    "Score 1.0 for perfect execution. Deduct 0.33 for failing to explain the root cause, and 0.33 for a poor complexity fix. Output ONLY the float."
                )
            }
        ]