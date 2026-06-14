"""
Genetic_algorithm_processes/S1_selection/methods/fitness/fitness_function.py
"""

from .accuracy import Accuracy
from .speed import Speed
from .token_limit import Token_limit_ratio

class FitnessCalculation:
    def __init__(self,
        accuracy_weight: float = 1.0,
        speed_weight: float = 0.5,
        token_limit_weight: float = 0.5,
        accuracy_instance = Accuracy(),
        token_limit_instance = Token_limit_ratio(),
        speed_instance = Speed(),
    ):
        self.accuracy_weight = accuracy_weight
        self.speed_weight = speed_weight
        self.token_limit_weight = token_limit_weight

        self.accuracy_instance = accuracy_instance
        self.token_limit_instance = token_limit_instance
        self.speed_instance = speed_instance
    
    def evaluate_prompt_chain(self, prompt_chain: list[tuple], prompt_output_chain: list[tuple], initial_input: str, solution_output: str) -> tuple[float, dict]:
        """
        Returns:
        - fitness_score (float)
        - telemetry (dict) containing raw stats for the dashboard
        """
        # Accuracy score
        accuracy_score = self.accuracy_instance.get_accuracy_score(prompt_output_chain[-1][0], initial_input, solution_output)

        # Token limit score & total tokens
        token_limit_score = 1.0
        total_tokens = 0
        for i in range(len(prompt_chain)):
            model_name = prompt_chain[i][0]
            prompt_token_used = prompt_output_chain[i][1].get('prompt_eval_count', 0)
            eval_token_used = prompt_output_chain[i][1].get('eval_count', 0)
            
            total_tokens += (prompt_token_used + eval_token_used)
            token_limit_score *= self.token_limit_instance.get_token_limit_score(prompt_token_used, eval_token_used, model_name)

        # Speed score & total time
        sum_total_duration = sum(step[1].get('total_duration', 0.0) for step in prompt_output_chain)
        speed_score = self.speed_instance.calculate_speed_score(sum_total_duration)

        # Combine scores into fitness
        fitness_score = (accuracy_score * self.accuracy_weight) * (speed_score ** self.speed_weight) * (token_limit_score ** self.token_limit_weight)
        
        # Package the raw data for the dashboard
        telemetry = {
            "accuracy": accuracy_score,
            "time_taken": sum_total_duration,
            "token_usage": total_tokens
        }
        
        return fitness_score, telemetry
    
    def evaluate_population(self, evaluated_chains: list[tuple], initial_input: str, solution_output: str) -> list[tuple[list[tuple], float, dict]]:
        population_fitness = []
        
        for prompt_chain, prompt_output_chain in evaluated_chains:
            fitness_score, telemetry = self.evaluate_prompt_chain(
                prompt_chain,
                prompt_output_chain,
                initial_input,
                solution_output
            )
            population_fitness.append((prompt_chain, fitness_score, telemetry))
            
        return population_fitness