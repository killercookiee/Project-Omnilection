# Input ratio of used tokens to total token limit (float between 0 and 1)
# Output a normalized token limit score between 0 and 1
# The function returns 0 if the ratio is close to 0, otherwise returns 1

# Customizable parameters:
# hard limit threshold: threshold below which the ratio is considered close to zero                 | default is 0.0
# soft limit margin: margin above the hard limit threshold where the score transitions from 0 to 1  | default is 0.1
# transition type: function defining how the score transitions from 0 to 1 within the soft margin   | default is smoothstep


from LLM_models.model_registry import Model_Registry

class Token_limit_ratio:
    def __init__(self, hard_limit_threshold=0.0, soft_limit_margin=0.1):
        self.hard_limit_threshold = hard_limit_threshold
        self.soft_limit_margin = soft_limit_margin

    def approximate_token_usage(self, text):
        """Approximate token usage based on text length / 4."""
        input_length = len(text)
        token_used = input_length // 4
        return token_used

    def calculated_tokens(self, prompt_input, model_name):
        """Receives prompt history and model_name to compute token usage and limit."""
        max_tokens = Model_Registry[model_name]["max_context_tokens"]
        token_used = self.approximate_token_usage(prompt_input)
        return token_used, max_tokens

    def token_limit_ratio_score(self, token_used, token_limit):
        """
        Token limit function to normalize token usage ratio into a score between 0 and 1.
        
        Parameters:
        - input_ratio: The ratio of used tokens to total token limit (float between 0 and 1)
        - hard_limit_threshold: Threshold below which the ratio is considered close to zero (float)
        - soft_limit_margin: Margin above the hard limit threshold for smooth transition (float)
        
        Returns:
        - limited_score: The normalized token limit score (float)
        """
        token_ratio = token_used / token_limit if token_limit > 0 else 0.0

        if token_ratio <= self.hard_limit_threshold:
            return 0.0
        elif token_ratio >= self.soft_limit_margin:
            return 1.0
        else:
            # Smoothstep transition
            x = (token_ratio - self.hard_limit_threshold) / self.soft_limit_margin
            return x * x * (3 - 2 * x)
        
    def get_token_limit_score(self, prompt_input, model_name):
        """Calculate the token limit score based on prompt input and model name."""
        token_used, token_limit = self.calculated_tokens(prompt_input, model_name)
        return self.token_limit_ratio_score(token_used, token_limit)
        
    
        
if __name__ == "__main__":
    token_limit_instance = Token_limit_ratio(hard_limit_threshold=0.0, soft_limit_margin=0.1)

    prompt_input = "This is a sample prompt input to test token limit ratio scoring."
    model_name = "gpt-3.5-turbo"
    score = token_limit_instance.get_token_limit_score(prompt_input, model_name)
    print(f"Token Limit Score: {score}")