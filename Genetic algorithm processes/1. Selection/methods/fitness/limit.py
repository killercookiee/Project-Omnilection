# Input score (float) for example token limit and speed
# Output a non-normalized score >= 0
# The function multiplies the input_score by 0 if the score is close to zero, otherwise mutliplies by 1

# Customizable parameters:
# hard limit threshold: threshold below which the score is considered close to zero                 | default is 0
# soft limit margin: margin above the hard limit threshold where the score transitions from 0 to 1  | default is 0.1
# transition type: function defining how the score transitions from 0 to 1 within the soft margin   | default is smoothstep


def limit(input_score, hard_limit_threshold=0.0, soft_limit_margin=0.1):
    """
    Limit function to cap scores between 0 and 1 based on thresholds.
    
    Parameters:
    - input_score: The score to be limited (float)
    - hard_limit_threshold: Threshold below which the score is considered close to zero (float)
    - soft_limit_margin: Margin above the hard limit threshold for smooth transition (float)
    
    Returns:
    - limited_score: The limited score (float)
    """
    if input_score <= hard_limit_threshold:
        return 0.0
    elif input_score >= soft_limit_margin:
        return input_score
    else:
        # Smoothstep transition
        x = (input_score - hard_limit_threshold) / soft_limit_margin
        return x * x * (3 - 2 * x)