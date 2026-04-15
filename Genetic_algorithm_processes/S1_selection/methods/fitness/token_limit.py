# Input ratio of used tokens to total token limit (float between 0 and 1)
# Output a normalized token limit score between 0 and 1
# The function returns 1 if the ratio is low, and 0 if the ratio is high (near limit)

# Customizable parameters:
# soft_limit_threshold: threshold below which the score is 1 (plenty of tokens available)    | default is 0.9
# hard_limit_threshold: threshold above which the score is 0 (at token limit)                | default is 1.0
# transition type: function defining how the score transitions from 1 to 0 within the margin | default is smoothstep

"""
Genetic_algorithm_processes/S1_selection/methods/fitness/token_limit.py
"""

import json
import os


class Token_limit_ratio:
    def __init__(self,
        hard_limit_threshold: float = 1.0,
        soft_limit_threshold: float = 0.85,
        verbose: bool = False
    ):
        """
        Parameters:
        - hard_limit_threshold: ratio at which score becomes 0 (at or above this ratio means token limit reached)
        - soft_limit_threshold: ratio below which score is 1 (below this ratio means plenty of tokens available)
        - verbose: enable pretty printing of scores and thresholds
        """
        self.soft_limit_threshold = soft_limit_threshold
        self.hard_limit_threshold = hard_limit_threshold
        self.verbose = verbose

        if self.verbose:
            print(f"[Token_limit_ratio] Initialized — soft: {self.soft_limit_threshold} | hard: {self.hard_limit_threshold}")

        file_path = 'LLM_models/model_registry.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                self.model_registry = json.load(file)
            if self.verbose:
                print(f"[Token_limit_ratio] ✅ Registry loaded from {file_path}")
        else:
            print(f"[Token_limit_ratio] ❌ ERROR: Could not find registry at {file_path}")
            self.model_registry = {}

    def calculated_tokens(self, prompt_token_used: int, eval_token_used: int, model_name: str) -> tuple[int, int]:
        """Receives prompt history and model_name to compute token usage and limit."""
        max_tokens = self.model_registry[model_name]["context_length"]
        token_used = prompt_token_used + eval_token_used

        if self.verbose:
            print(f"[Token_limit_ratio] Model: {model_name} | Used: {token_used} tokens / {max_tokens} limit")

        return token_used, max_tokens

    def token_limit_ratio_score(self, token_used: int, token_limit: int) -> float:
        token_ratio = token_used / token_limit if token_limit > 0 else 0.0

        if token_ratio <= self.soft_limit_threshold:
            score = 1.0
        elif token_ratio >= self.hard_limit_threshold:
            score = 0.0
        else:
            x = (token_ratio - self.soft_limit_threshold) / (self.hard_limit_threshold - self.soft_limit_threshold)
            score = 1.0 - (x * x * (3 - 2 * x))

        if self.verbose:
            status = "✅" if score >= 0.75 else "⚠️" if score > 0.0 else "❌"
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"\n{'─'*40}")
            print(f"  {status}  Token limit evaluation")
            print(f"      Ratio  : {token_ratio:.1%}  (soft: {self.soft_limit_threshold:.0%} | hard: {self.hard_limit_threshold:.0%})")
            print(f"      Score  : {score:.3f}  [{bar}]")
            print(f"{'─'*40}\n")

        return score

    def get_token_limit_score(self, prompt_token_used: int, eval_token_used: int, model_name: str) -> float:
        """Calculate the token limit score based on prompt input and model name."""
        try:
            token_used, token_limit = self.calculated_tokens(prompt_token_used, eval_token_used, model_name)
            return self.token_limit_ratio_score(token_used, token_limit)
        except Exception as e:
            print(f"[Token_limit_ratio] ❌ ERROR for model {model_name}: {e} — returning 0.0")
            return 0.0


if __name__ == "__main__":
    token_limit_instance = Token_limit_ratio(hard_limit_threshold=1.0, soft_limit_threshold=0.85, verbose=True)

    score = token_limit_instance.get_token_limit_score(
        prompt_token_used=1400,
        eval_token_used=300,
        model_name="smollm:135m"
    )
    print(f"Final — Token Limit Score: {score:.3f}")