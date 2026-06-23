"""
Genetic_algorithm_processes/S1_selection/methods/fitness/_model_cost.py
"""

# Input model cost (float) in terms of money and compute resources (e.g. latency, energy consumption, ram usage, etc.)
# Output a normalized cost score between 0 and 1
# The function is an inverse exponential scaling from (0, cost_punishment) to (infinity, 0), function is 1 at cost 0 to encourage free models

# Customizable parameters:
# scaling factor: controls the steepness of the curve (float)               | default is 0.1
# cost_punishment: maximum score when the model is not free                 | default is 0.5


class ModelCost:
    def __init__(self, scaling_factor=0.1, cost_punishment=0.5):
        self.scaling_factor = scaling_factor
        self.cost_punishment = cost_punishment

    def get_model_cost(self, model):
        """Receives model and computes its cost in dollars."""
        model_cost_value = 0
        return model_cost_value

    def model_cost(self, input_cost, scaling_factor=0.1, cost_punishment=0.5):
        """
        Model cost function to normalize model cost scores between 0 and 1.
        
        Parameters:
        - input_cost: The model cost (float)
        - scaling_factor: Controls the steepness of the curve (float)
        - cost_punishment: Maximum score when the model is not free (float)
        
        Returns:
        - normalized_score: The normalized cost score (float)
        """
        if input_cost <= 0:
            return 1.0
        else:
            import math
            normalized_score = cost_punishment * math.exp(-scaling_factor * input_cost)
            return normalized_score
        
if __name__ == "__main__":
    model_cost_instance = ModelCost(scaling_factor=0.1, cost_punishment=0.5)

    models = [
        "gpt-3.5-turbo",   # Free model
        "gpt-4",           # Paid model
        "custom-model"     # Hypothetical paid model
    ]
    for model in models:
        model_cost_value = model_cost_instance.get_model_cost(model)
        cost_score = model_cost_instance.model_cost(model_cost_value)
        print(f"Computed Model Cost Score for '{model}': {cost_score}")