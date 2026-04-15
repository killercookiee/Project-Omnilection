# Input time taken (float) for example in seconds
# Output a normalized speed score between 0 and 1
# The function is a smoothstep function between (0, 1) and (speed_limit, 0)

# Customizable parameters:
# speed limit: maximum time taken for score to be non zero (float)          | default is 10 seconds

"""
Genetic_algorithm_processes/S1_selection/methods/fitness/speed.py
"""

import time

class Speed:
    def __init__(self,
                 time_limit: float = 15.0,
                 sharpness: float = 0.75,
                 inflection_point: float = 0.8,
                 verbose: bool = False):
        """
        Parameters:
        - time_limit: maximum time before score is 0
        - sharpness: 0 = linear, 1 = sharp S-curve (controls smoothstep polynomial order)
        - inflection_point: 0 = curve bends early, 1 = curve bends late (as fraction of time_limit)
        - verbose: enable pretty printing
        """
        self.speed_limit = time_limit
        self.sharpness = max(0.0, min(1.0, sharpness))
        self.inflection_point = max(0.01, min(0.99, inflection_point))
        self.verbose = verbose

        if self.verbose:
            print(f"[Speed] Initialized — limit: {self.speed_limit}s | sharpness: {self.sharpness} | inflection: {self.inflection_point}")

    def _curve(self, x: float) -> float:
        """
        Blends between linear (sharpness=0) and a high-order smoothstep (sharpness=1).
        Inflection point shifts where the curve bends by warping x before applying smoothstep.
        """
        # Warp x so the inflection lands at the desired point
        if x < self.inflection_point:
            x_warped = 0.5 * (x / self.inflection_point)
        else:
            x_warped = 0.5 + 0.5 * ((x - self.inflection_point) / (1.0 - self.inflection_point))

        # Blend between linear and smoothstep based on sharpness
        # Higher order smoothstep (N=1 is classic, higher = sharper shoulders)
        order = 1 + round(self.sharpness * 4)  # order 1–5
        smoothstep = self._smoothstep(x_warped, order)
        linear = x_warped

        return linear + self.sharpness * (smoothstep - linear)

    def _smoothstep(self, x: float, order: int) -> float:
        """Generalized smoothstep of given order (Ken Perlin formulation)."""
        for _ in range(order):
            x = x * x * (3 - 2 * x)
        return x

    def calculate_speed_score(self, input_time: float) -> float:
        if input_time >= self.speed_limit:
            score = 0.0
        elif input_time <= 0.0:
            score = 1.0
        else:
            x = input_time / self.speed_limit
            score = 1.0 - self._curve(x)

        if self.verbose:
            status = "✅" if score >= 0.75 else "⚠️" if score > 0.0 else "❌"
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"\n{'─'*40}")
            print(f"  {status}  Speed evaluation")
            print(f"      Time   : {input_time:.3f}s / {self.speed_limit}s limit")
            print(f"      Score  : {score:.3f}  [{bar}]")
            print(f"{'─'*40}\n")

        return score


if __name__ == "__main__":
    speed_instance = Speed(time_limit=10.0, sharpness=0.75, inflection_point=0.8, verbose=True)

    start_time = time.time()
    time.sleep(3)
    elapsed_time = time.time() - start_time

    score = speed_instance.calculate_speed_score(elapsed_time)
    print(f"Final — Elapsed: {elapsed_time:.3f}s | Score: {score:.3f}")