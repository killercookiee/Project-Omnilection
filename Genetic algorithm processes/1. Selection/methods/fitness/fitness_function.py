# Input of scores from accuracy, limit, model cost, token limit, speed functions
# Combine these scores into a single fitness score for selection

# Fitness = (alpha * accuracy_score + beta * model_cost_score) * limit(gamma * token_limit_score) * limit( delta * speed_score)

# Customizable parameters:
# alpha, beta, gamma, delta: weights for each component in the fitness calculation         | default is 1.0 for all


from accuracy import Accuracy
from speed import Speed
from token_limit import Token_limit_ratio
from limit import limit

class FitnessFunction:
    def __init__(self, accuracy_weight=1.0, speed_weight=1.0, token_limit_weight=1.0,
                 accuracy_scaling_factor=0.05, accuracy_testing_model=None,
                 speed_limit=10.0,
                 token_hard_limit_threshold=0.0, token_soft_limit_margin=0.1):
        self.alpha = accuracy_weight
        self.gamma = speed_weight
        self.delta = token_limit_weight

        # Instances
        self.limit_func = limit
        self.accuracy_instance = Accuracy(scaling_factor=accuracy_scaling_factor, testing_model=accuracy_testing_model)
        self.token_limit_instance = Token_limit_ratio(hard_limit_threshold=token_hard_limit_threshold, soft_limit_margin=token_soft_limit_margin)
        self.speed_instance = Speed(speed_limit=speed_limit)
    
    def get_fitness_score(self, model_context_list, final_prompt_output, elapsed_time):
        """
        Compute the overall fitness score based on model context, final output, and elapsed time.
        
        Parameters:
        - model_context_list: List of tuples containing (model_name, prompt_input)
        - final_prompt_output: The final output from the model
        - elapsed_time: Total time taken for the process (float)
        
        Returns:
        - fitness_score: The computed fitness score (float)
        """
        # Accuracy score
        accuracy_score = self.accuracy_instance.get_accuracy_score(final_prompt_output)

        # Token limit score
        token_limit_score = 1.0
        # We get the product of token limit scores across all model contexts
        for model_name, prompt_input in model_context_list:
            token_limit_score *= self.token_limit_instance.get_token_limit_score(prompt_input, model_name)

        # Speed score
        speed_score = self.speed_instance.calculate_speed_score(elapsed_time)

        # Combine scores into fitness
        fitness_score = (self.alpha * accuracy_score) * self.limit_func(self.gamma * speed_score) * self.limit_func(self.delta * token_limit_score)

        return fitness_score


if __name__ == "__main__":
    fitness_function = FitnessFunction(accuracy_weight=1.0, model_cost_weight=0.5, speed_weight=1.0, token_limit_weight=1.0)

    # Phenotype chain: [(model_name, prompt_input), ...]
    phenotype_chain = [("gpt-3.5-turbo", "This is an input prompt"), ("gpt-3", "This another input prompt")]
    def run_phenotype(phenotype_chain):
        # Output: [(prompt_output, time_taken), ...]
        return [("The capital of France is Paris.", 2.5), ("The capital of Germany is Berlin.", 3.0)]
    prompt_output_chain = run_phenotype(phenotype_chain)

    total_elapsed_time = sum([time_taken for _, time_taken in prompt_output_chain])
    final_output = prompt_output_chain[-1][0]
    fitness_score = fitness_function.get_fitness_score(phenotype_chain, final_output, total_elapsed_time)