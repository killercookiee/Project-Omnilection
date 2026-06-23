"""
Genetic_algorithm_processes/S1_selection/methods/fitness/fitness_function.py
"""

import time
import math
from Genetic_algorithm_processes.S1_selection.methods.fitness.accuracy import AccuracyCalculation

class FitnessCalculation:
    def __init__(self, accuracy_func=None, hyperparameters=None):
        self.accuracy_func = accuracy_func if accuracy_func else AccuracyCalculation().evaluate_accuracy
        
        # 🚨 THE HYPERPARAMETERS
        # Tune these to change evolutionary pressure! Must sum to 1.0
        self.weights = hyperparameters or {
            "accuracy_weight": 0.60,  # 60% of the score is strictly correctness
            "time_weight": 0.15,      # 15% is execution speed
            "token_weight": 0.15,     # 15% is prompt context cost
            "length_weight": 0.10     # 10% is keeping the chain short
        }
        
        assert math.isclose(sum(self.weights.values()), 1.0), "Fitness weights must sum exactly to 1.0"

    def calculate_time_penalty(self, time_taken: float, optimal_time: float = 2.0, max_time: float = 20.0) -> float:
        if time_taken <= optimal_time: return 1.0
        if time_taken >= max_time: return 0.0
        return max(0.0, math.exp(-0.2 * (time_taken - optimal_time)))

    def calculate_length_penalty(self, chain_length: int, optimal_length: int = 2) -> float:
        if chain_length <= optimal_length: return 1.0
        return max(0.0, 1.0 - (0.1 * (chain_length - optimal_length)))

    def calculate_token_penalty(self, total_tokens: int, optimal_tokens: int = 150, max_tokens: int = 1500) -> float:
        if total_tokens <= optimal_tokens: return 1.0
        if total_tokens >= max_tokens: return 0.0
        return max(0.0, 1.0 - ((total_tokens - optimal_tokens) / (max_tokens - optimal_tokens)))

    def calculate_final_fitness(self, accuracy: float, time_score: float, length_score: float, token_score: float) -> float:
        # Gatekeeper: If it's totally wrong, efficiency doesn't matter. Kill it.
        if accuracy <= 0.05:
            return 0.0
            
        # Calculate the composite efficiency bonus (scaled 0.0 to 1.0)
        efficiency_sum = (
            (time_score * self.weights["time_weight"]) +
            (length_score * self.weights["length_weight"]) +
            (token_score * self.weights["token_weight"])
        )
        
        # Normalize the efficiency sum based on the remaining weight pie
        total_efficiency_weight = self.weights["time_weight"] + self.weights["length_weight"] + self.weights["token_weight"]
        normalized_efficiency = efficiency_sum / total_efficiency_weight
        
        # Final Formula: Weighted accuracy + (accuracy-scaled efficiency bonus)
        # We multiply efficiency by accuracy so a "half-right" answer doesn't get 
        # a massive boost just because it was super fast.
        base_score = accuracy * self.weights["accuracy_weight"]
        efficiency_bonus = (normalized_efficiency * accuracy) * (1.0 - self.weights["accuracy_weight"])
        
        return base_score + efficiency_bonus

    def evaluate_prompt_chain(self, chain: list, runner, initial_input: str, solution_output: str) -> tuple[float, dict]:
        start_time = time.time()
        prompt_output_chain = runner.run_prompt_chain(chain, initial_input)
        time_taken = time.time() - start_time

        if not prompt_output_chain or not isinstance(prompt_output_chain[-1], (list, tuple)) or len(prompt_output_chain[-1]) == 0:
             return 0.0, {"time_taken": time_taken, "accuracy": 0.0, "error": "Malformed output"}

        final_output = str(prompt_output_chain[-1][0])
        accuracy = self.accuracy_func(solution_output, final_output, initial_input=initial_input, runner=runner)

        chain_length = len(chain)
        total_tokens_used = 0
        for step_result in prompt_output_chain:
            if isinstance(step_result, (list, tuple)) and len(step_result) > 1:
                metrics = step_result[1]
                if isinstance(metrics, dict):
                    prompt_tokens = metrics.get("prompt_eval_count", 0)
                    eval_tokens = metrics.get("eval_count", 0)
                    total_tokens_used += (prompt_tokens + eval_tokens)

        # In this new architecture, these are "Scores" (1.0 = perfect), not "Penalties"
        time_score = self.calculate_time_penalty(time_taken)
        length_score = self.calculate_length_penalty(chain_length)
        token_score = self.calculate_token_penalty(total_tokens_used)

        final_fitness = self.calculate_final_fitness(accuracy, time_score, length_score, token_score)

        telemetry = {
            "time_taken": time_taken,
            "accuracy": accuracy,
            "chain_length": chain_length,
            "total_tokens_used": total_tokens_used,
            "time_score": time_score,
            "length_score": length_score,
            "token_score": token_score,
            "final_output": final_output
        }

        return final_fitness, telemetry