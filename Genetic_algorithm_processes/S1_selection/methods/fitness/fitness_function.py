"""
Genetic_algorithm_processes/S1_selection/methods/fitness/fitness_function.py
"""

import time
import math
from Genetic_algorithm_processes.S1_selection.methods.fitness.accuracy import AccuracyCalculation

class FitnessCalculation:
    def __init__(self, accuracy_func=None):
        # We pass the class method so it can be called dynamically
        self.accuracy_func = accuracy_func if accuracy_func else AccuracyCalculation().evaluate_accuracy

    def calculate_time_penalty(self, time_taken: float, optimal_time: float = 2.0, max_time: float = 20.0) -> float:
        if time_taken <= optimal_time:
            return 1.0
        if time_taken >= max_time:
            return 0.1
        penalty = math.exp(-0.2 * (time_taken - optimal_time))
        return max(0.1, penalty)

    def calculate_length_penalty(self, chain_length: int, optimal_length: int = 2) -> float:
        if chain_length <= optimal_length:
            return 1.0
        penalty = 1.0 - (0.1 * (chain_length - optimal_length))
        return max(0.3, penalty)

    def calculate_token_penalty(self, total_tokens: int, optimal_tokens: int = 150, max_tokens: int = 1500) -> float:
        if total_tokens <= optimal_tokens:
            return 1.0
        if total_tokens >= max_tokens:
            return 0.2
        penalty = 1.0 - 0.8 * ((total_tokens - optimal_tokens) / (max_tokens - optimal_tokens))
        return max(0.2, penalty)

    def calculate_final_fitness(self, accuracy: float, time_penalty: float, length_penalty: float, token_penalty: float) -> float:
        # Heavily weight accuracy: if accuracy is 0, the whole score is 0
        return accuracy * time_penalty * length_penalty * token_penalty

    def evaluate_prompt_chain(self, chain: list, runner, initial_input: str, solution_output: str) -> tuple[float, dict]:
        start_time = time.time()
        prompt_output_chain = runner.run_prompt_chain(chain, initial_input)
        time_taken = time.time() - start_time

        if not prompt_output_chain or not isinstance(prompt_output_chain[-1], (list, tuple)) or len(prompt_output_chain[-1]) == 0:
             return 0.0, {"time_taken": time_taken, "accuracy": 0.0, "error": "Malformed output"}

        final_output = str(prompt_output_chain[-1][0])

        # 🚨 THE FIX: Pass the initial_input and runner to the accuracy function!
        accuracy = self.accuracy_func(solution_output, final_output, initial_input=initial_input, runner=runner)

        chain_length = len(chain)
        total_tokens_used = 0
        for step_result in prompt_output_chain:
            if isinstance(step_result, (list, tuple)) and len(step_result) > 1:
                metrics = step_result[1]
                if isinstance(metrics, dict):
                    # Ollama typically returns prompt_eval_count (input) and eval_count (output)
                    prompt_tokens = metrics.get("prompt_eval_count", 0)
                    eval_tokens = metrics.get("eval_count", 0)
                    total_tokens_used += (prompt_tokens + eval_tokens)

        time_penalty = self.calculate_time_penalty(time_taken)
        length_penalty = self.calculate_length_penalty(chain_length)
        token_penalty = self.calculate_token_penalty(total_tokens_used)

        final_fitness = self.calculate_final_fitness(accuracy, time_penalty, length_penalty, token_penalty)

        telemetry = {
            "time_taken": time_taken,
            "accuracy": accuracy,
            "chain_length": chain_length,
            "total_tokens_used": total_tokens_used,
            "time_penalty": time_penalty,
            "length_penalty": length_penalty,
            "token_penalty": token_penalty,
            "final_output": final_output
        }

        return final_fitness, telemetry