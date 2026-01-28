# Input a accuracy score (percentage between 0 and 100)
# Output a normalized accuracy score between 0 and 1
# The function is a exponential scaling function from (0,0) to (100, 1)

# Customizable parameters:
# scaling factor: controls the steepness of the curve (float)               | default is 0.05


def accuracy(input_score, scaling_factor=0.05):
    """
    Accuracy function to normalize accuracy scores between 0 and 1.
    
    Parameters:
    - input_score: The accuracy score (float between 0 and 100)
    - scaling_factor: Controls the steepness of the curve (float)
    
    Returns:
    - normalized_score: The normalized accuracy score (float)
    """
    if input_score < 0:
        return 0.0
    elif input_score > 100:
        return 1.0
    else:
        import math
        normalized_score = 1 - math.exp(-scaling_factor * input_score)
        return normalized_score