"""
Genetic_algorithm_processes/S1_selection/methods/fitness/speed.py
"""

import time

class Speed:
    def __init__(self,
                 time_limit: float = 60.0,     # Increased to 60s for local LLMs
                 sharpness: float = 0.75,
                 inflection_point: float = 0.8,
                 min_score: float = 0.2,       # Floor to prevent instant death
                 verbose: bool = False):
        self.speed_limit = time_limit
        self.sharpness = max(0.0, min(1.0, sharpness))
        self.inflection_point = max(0.01, min(0.99, inflection_point))
        self.min_score = max(0.0, min(1.0, min_score))
        self.verbose = verbose

        if self.verbose:
            print(f"[Speed] Initialized — limit: {self.speed_limit}s | floor: {self.min_score}")

    def _curve(self, x: float) -> float:
        if x < self.inflection_point:
            x_warped = 0.5 * (x / self.inflection_point)
        else:
            x_warped = 0.5 + 0.5 * ((x - self.inflection_point) / (1.0 - self.inflection_point))

        order = 1 + round(self.sharpness * 4)  
        smoothstep = self._smoothstep(x_warped, order)
        linear = x_warped

        return linear + self.sharpness * (smoothstep - linear)

    def _smoothstep(self, x: float, order: int) -> float:
        for _ in range(order):
            x = x * x * (3 - 2 * x)
        return x

    def calculate_speed_score(self, input_time: float) -> float:
        if input_time >= self.speed_limit:
            score = self.min_score
        elif input_time <= 0.0:
            score = 1.0
        else:
            x = input_time / self.speed_limit
            base_score = 1.0 - self._curve(x)
            # Squeeze the score between min_score and 1.0
            score = self.min_score + base_score * (1.0 - self.min_score)

        if self.verbose:
            status = "✅" if score >= 0.75 else "⚠️" if score > self.min_score else "❌"
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"\n{'─'*40}")
            print(f"  {status}  Speed evaluation")
            print(f"      Time   : {input_time:.3f}s / {self.speed_limit}s limit")
            print(f"      Score  : {score:.3f}  [{bar}]")
            print(f"{'─'*40}\n")

        return score