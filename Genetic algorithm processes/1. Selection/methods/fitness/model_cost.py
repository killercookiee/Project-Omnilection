# Input model cost (float) in terms of money
# Output a normalized cost score between 0 and 1
# The function is an inverse exponential scaling from (0, cost_punishment) to (infinity, 0), function is 1 at cost 0 to encourage free models

# Customizable parameters:
# scaling factor: controls the steepness of the curve (float)               | default is 0.1
# cost_punishment: maximum score when the model is not free                 | default is 0.5


def model_cost(input_cost, scaling_factor=0.1, cost_punishment=0.5):
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