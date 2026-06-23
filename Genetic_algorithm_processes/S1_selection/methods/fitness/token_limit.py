"""
Genetic_algorithm_processes/S1_selection/methods/fitness/token_limit.py
"""

import json
import os

class Token_limit_ratio:
    def __init__(self,
        hard_limit_threshold: float = 1.0,
        soft_limit_threshold: float = 0.85,
        min_score: float = 0.2,             # Floor to prevent instant death
        verbose: bool = False
    ):
        self.soft_limit_threshold = soft_limit_threshold
        self.hard_limit_threshold = hard_limit_threshold
        self.min_score = min_score
        self.verbose = verbose

        file_path = 'LLM_models/model_registry.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                self.model_registry = json.load(file)
        else:
            self.model_registry = {}

    def calculated_tokens(self, prompt_token_used: int, eval_token_used: int, model_name: str) -> tuple[int, int]:
        max_tokens = self.model_registry.get(model_name, {}).get("context_length", 2048)
        token_used = prompt_token_used + eval_token_used
        return token_used, max_tokens

    def token_limit_ratio_score(self, token_used: int, token_limit: int) -> float:
        token_ratio = token_used / token_limit if token_limit > 0 else 0.0

        if token_ratio <= self.soft_limit_threshold:
            score = 1.0
        elif token_ratio >= self.hard_limit_threshold:
            score = self.min_score
        else:
            x = (token_ratio - self.soft_limit_threshold) / (self.hard_limit_threshold - self.soft_limit_threshold)
            base_score = 1.0 - (x * x * (3 - 2 * x))
            # Squeeze the score between min_score and 1.0
            score = self.min_score + base_score * (1.0 - self.min_score)

        return score

    def get_token_limit_score(self, prompt_token_used: int, eval_token_used: int, model_name: str) -> float:
        try:
            token_used, token_limit = self.calculated_tokens(prompt_token_used, eval_token_used, model_name)
            return self.token_limit_ratio_score(token_used, token_limit)
        except Exception:
            return self.min_score