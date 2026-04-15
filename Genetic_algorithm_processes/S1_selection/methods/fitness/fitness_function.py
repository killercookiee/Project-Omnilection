"""
Genetic_algorithm_processes/S1_selection/methods/fitness/fitness_function.py
"""

# Input of scores from accuracy, limit, model cost, token limit, speed functions
# Combine these scores into a single fitness score for selection

# Fitness = (alpha * accuracy_score + beta * model_cost_score) * limit(gamma * token_limit_score) * limit( delta * speed_score)

# Customizable parameters:
# alpha, beta, gamma, delta: weights for each component in the fitness calculation         | default is 1.0 for all


from Genetic_algorithm_processes.ollama_run import PromptChainRunner
from Genetic_algorithm_processes.S1_selection.methods.evaluation.compare_solution_eval import CompareSolutionEval

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
        """
        Parameters:
        - accuracy_weight: weight for the accuracy score in the fitness calculation (float)
        - speed_weight: weight for the speed score in the fitness calculation (float)
        - token_limit_weight: weight for the token limit score in the fitness calculation (float)
        - accuracy_instance: instance of the Accuracy class to compute accuracy scores
        - token_limit_instance: instance of the Token_limit_ratio class to compute token limit scores
        - speed_instance: instance of the Speed class to compute speed scores
        """
        self.accuracy_weight = accuracy_weight
        self.speed_weight = speed_weight
        self.token_limit_weight = token_limit_weight

        self.accuracy_instance = accuracy_instance
        self.token_limit_instance = token_limit_instance
        self.speed_instance = speed_instance
    
    def evaluate_prompt_chain(self, prompt_chain: list[list[tuple]], prompt_output_chain: list[tuple[str, dict[str, float]]], initial_input: str, solution_output: str) -> float:
        """
        Compute the overall fitness score based on model context, final output, and elapsed time.
        
        Parameters:
        - prompt_chain: List of sets containing (model_name, prompt_input_segment_1, prompt_input_segment_2, ... for each step in the chain
        - prompt_output_chain: List of tuples containing (prompt_output, model_run_info), where model_run_info includes total_duration, load_duration, prompt_eval_count, prompt_eval_duration, prompt_eval_rate, eval_count, eval_duration, eval_rate for each step in the chain
        
        Returns:
        - fitness_score: The computed fitness score (float)
        """
        # Accuracy score
        accuracy_score = self.accuracy_instance.get_accuracy_score(prompt_output_chain[-1][0], initial_input, solution_output)  # Use final output for accuracy evaluation

        # Token limit score
        token_limit_score = 1.0
        # We get the product of token limit scores across all model contexts
        for i in range(len(prompt_chain)):
            model_name = prompt_chain[i][0]
            prompt_token_used = prompt_output_chain[i][1].get('prompt_eval_count', 0)
            eval_token_used = prompt_output_chain[i][1].get('eval_count', 0)
            token_limit_score *= self.token_limit_instance.get_token_limit_score(prompt_token_used, eval_token_used, model_name)

        # Speed score
        # total time is sum of total_duration across all steps in the chain
        sum_total_duration = sum(step[1].get('total_duration', 0.0) for step in prompt_output_chain)
        speed_score = self.speed_instance.calculate_speed_score(sum_total_duration)

        # Combine scores into fitness
        fitness_score = (accuracy_score * self.accuracy_weight) * (speed_score ** self.speed_weight) * (token_limit_score ** self.token_limit_weight)
        return fitness_score
    
    def evaluate_population(self, prompt_chain_population: list[list[tuple]], run_prompt_chain: callable, initial_input: str, solution_output: str) -> float:
        # Run the prompt chain and compute fitness score for each prompt chain in the population
        population_fitness = []
        
        for prompt_chain in prompt_chain_population:
            # Run prompt chain and get output and model run info for each step
            prompt_output_chain = run_prompt_chain(prompt_chain, initial_input)

            # Compute fitness score for the prompt chain
            fitness_score = self.evaluate_prompt_chain(
                prompt_chain,
                prompt_output_chain,
                initial_input,
                solution_output
            )

            population_fitness.append((prompt_chain, fitness_score))
        return population_fitness


if __name__ == "__main__":
    # Define population and initial input
    prompt_chain_population = [
        [("gemma3:270m", "Summarize the task and give a hint of the answer. ", "Make it concise and clear."),
         ("qwen2.5-coder:0.5b", "Provide 1 word answer only")],

        [("smollm:360m", "Summarize the task. ", "Give a hint of the completely wrong answer."),
         ("qwen2:0.5b", "Give a wrong answer. ", "Explain why the right answer is correct and the wrong answer is wrong.")],

        [("deepseek-coder:latest", "Another different prompt.")]
    ]
    initial_input = "What is the capital of France?"
    solution_output = "Paris"


    # ====== Get Fitness Score ======
    # Run prompt chain and get final output
    prompt_chain_runner = PromptChainRunner()
    prompt_output_chain = prompt_chain_runner.run_prompt_chain(prompt_chain_population[0], initial_input)
    final_output = prompt_output_chain[-1][0]

    # Initialize fitness function with accuracy instance using CompareSolutionEval
    fitness_function = FitnessCalculation()
    fitness_score = fitness_function.evaluate_prompt_chain(prompt_chain_population[0], prompt_output_chain, initial_input, solution_output)
    print(f"Fitness Score: {fitness_score}")
    print("Finished fitness evaluation for single prompt chain.\n\n")


    # ====== Run Fitness Evaluation ======
    fitness_function = FitnessCalculation()
    population_fitness = fitness_function.evaluate_population(prompt_chain_population, prompt_chain_runner.run_prompt_chain, initial_input, solution_output)
    print("Population Fitness Scores:")
    for prompt_chain, fitness_score in population_fitness:
        print(f"Prompt Chain: {prompt_chain}, Fitness Score: {fitness_score}\n")