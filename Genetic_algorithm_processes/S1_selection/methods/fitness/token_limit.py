# Input ratio of used tokens to total token limit (float between 0 and 1)
# Output a normalized token limit score between 0 and 1
# The function returns 1 if the ratio is low, and 0 if the ratio is high (near limit)

# Customizable parameters:
# soft_limit_threshold: threshold below which the score is 1 (plenty of tokens available)    | default is 0.9
# hard_limit_threshold: threshold above which the score is 0 (at token limit)                | default is 1.0
# transition type: function defining how the score transitions from 1 to 0 within the margin | default is smoothstep


from LLM_models.model_registry import Model_Registry

class Token_limit_ratio:
    def __init__(self, hard_limit_threshold=1.0, soft_limit_threshold=0.9):
        self.soft_limit_threshold = soft_limit_threshold
        self.hard_limit_threshold = hard_limit_threshold

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
        
        Returns 1 when token usage is low (below soft_limit_threshold)
        Returns 0 when token usage is high (above hard_limit_threshold)
        Smoothly transitions between them in the margin
        
        Parameters:
        - token_used: Number of tokens used
        - token_limit: Maximum token limit
        
        Returns:
        - limited_score: The normalized token limit score (float between 0 and 1)
        """
        token_ratio = token_used / token_limit if token_limit > 0 else 0.0

        if token_ratio <= self.soft_limit_threshold:
            return 1.0  # Plenty of tokens available
        elif token_ratio >= self.hard_limit_threshold:
            return 0.0  # At or over token limit
        else:
            # Smoothstep transition from 1 to 0
            x = (token_ratio - self.soft_limit_threshold) / (self.hard_limit_threshold - self.soft_limit_threshold)
            smoothstep = x * x * (3 - 2 * x)
            return 1.0 - smoothstep  # Invert so it goes from 1 to 0
        
    def get_token_limit_score(self, prompt_input, model_name):
        """Calculate the token limit score based on prompt input and model name."""
        token_used, token_limit = self.calculated_tokens(prompt_input, model_name)
        return self.token_limit_ratio_score(token_used, token_limit)
        
    
        
if __name__ == "__main__":
    token_limit_instance = Token_limit_ratio(hard_limit_threshold=1.0, soft_limit_threshold=0.9)

    prompt_input = "This is a sample prompt input to test token limit ratio scoring."
    model_name = "gpt-3.5-turbo"
    score = token_limit_instance.get_token_limit_score(prompt_input, model_name)
    print(f"Token Limit Score: {score}")