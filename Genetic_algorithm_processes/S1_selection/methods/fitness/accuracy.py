# Input a accuracy score (percentage between 0 and 100)
# Output a normalized accuracy score between 0 and 1
# The function is a exponential scaling function from (0,0) to (100, 1)

# Customizable parameters:
# scaling_factor: controls the steepness of the curve (float)               | default is 0.05


"""
Genetic_algorithm_processes/S1_selection/methods/fitness/accuracy.py
"""

from Genetic_algorithm_processes.S1_selection.methods.evaluation.compare_solution_eval import CompareSolutionEval


class Accuracy:
    def __init__(self,
        scaling_factor: float = 0.45,
        evaluation_method=CompareSolutionEval(),
        verbose: bool = False
    ):
        """
        Parameters:
        - scaling_factor: controls the steepness of the curve (float)
        - evaluation_method: instance of a class that implements an evaluate(prompt_output, context) method
        - verbose: enable pretty printing of scores and thresholds
        """
        self.scaling_factor = scaling_factor
        self.evaluation_method = evaluation_method
        self.verbose = verbose

        if self.verbose:
            print(f"[Accuracy] Initialized — scaling factor: {self.scaling_factor}")

    def normalize_accuracy_score(self, accuracy_value: float) -> float:
        if accuracy_value < 0:
            score = 0.0
        elif accuracy_value > 100:
            score = 1.0
        else:
            c, n = 25, 3
            score = (accuracy_value / 100) ** (1 + c * self.scaling_factor ** n)

        if self.verbose:
            status = "✅" if score >= 0.75 else "⚠️" if score > 0.0 else "❌"
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"\n{'─'*40}")
            print(f"  {status}  Accuracy normalization")
            print(f"      Raw    : {accuracy_value:.1f}/100")
            print(f"      Score  : {score:.3f}  [{bar}]")
            print(f"{'─'*40}\n")

        return score

    def get_accuracy_score(self, prompt_output: str, initial_input: str, solution_output: str) -> float:
        if self.verbose:
            print(f"[Accuracy] Evaluating output against solution...")

        accuracy_value = self.evaluation_method.evaluate(prompt_output, {
            "initial_input": initial_input,
            "solution_output": solution_output
        })
        score = self.normalize_accuracy_score(accuracy_value)

        if self.verbose:
            print(f"[Accuracy] Final score: {score:.3f}  (raw eval: {accuracy_value:.1f}/100)")

        return score


if __name__ == "__main__":
    accuracy_instance = Accuracy(scaling_factor=0.45, verbose=True)

    score = accuracy_instance.get_accuracy_score(
        prompt_output="Moscow is the capital of Russia. St. Petersburg is a minor city.",
        initial_input="What is the capital of Russia?",
        solution_output="Moscow is the capital of Russia. St. Petersburg is a major city but not the capital."
    )
    print(f"Final — Accuracy Score: {score:.3f}")