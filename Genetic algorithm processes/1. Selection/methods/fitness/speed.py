# Input time taken (float) for example in seconds
# Output a normalized speed score between 0 and 1
# The function is a smoothstep function between (0, 1) and (speed_limit, 0)

# Customizable parameters:
# speed limit: maximum time taken for score to be non zero (float)          | default is 10 seconds


def speed(input_time, speed_limit=10.0):
    """
    Speed function to normalize time taken into a score between 0 and 1.
    
    Parameters:
    - input_time: The time taken (float)
    - speed_limit: Maximum time for non-zero score (float)
    
    Returns:
    - speed_score: The normalized speed score (float)
    """
    if input_time >= speed_limit:
        return 0.0
    else:
        # Smoothstep function
        x = input_time / speed_limit
        return 1 - (3 * x**2 - 2 * x**3)  # Smoothstep formula