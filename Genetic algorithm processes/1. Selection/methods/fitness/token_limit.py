# Input ratio of used tokens to total token limit (float between 0 and 1)
# Output a normalized token limit score between 0 and 1
# The function returns 0 if the ratio is close to 0, otherwise returns 1

# Customizable parameters:
# hard limit threshold: threshold below which the ratio is considered close to zero                 | default is 0.0
# soft limit margin: margin above the hard limit threshold where the score transitions from 0 to 1  | default is 0.1
# transition type: function defining how the score transitions from 0 to 1 within the soft margin   | default is smoothstep


def token_limit(input_ratio, hard_limit_threshold=0.0, soft_limit_margin=0.1):
    """
    Token limit function to normalize token usage ratio into a score between 0 and 1.
    
    Parameters:
    - input_ratio: The ratio of used tokens to total token limit (float between 0 and 1)
    - hard_limit_threshold: Threshold below which the ratio is considered close to zero (float)
    - soft_limit_margin: Margin above the hard limit threshold for smooth transition (float)
    
    Returns:
    - limited_score: The normalized token limit score (float)
    """
    if input_ratio <= hard_limit_threshold:
        return 0.0
    elif input_ratio >= soft_limit_margin:
        return 1.0
    else:
        # Smoothstep transition
        x = (input_ratio - hard_limit_threshold) / soft_limit_margin
        return x * x * (3 - 2 * x)