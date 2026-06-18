"""
Genetic_algorithm_processes/S1_selection/methods/fitness/accuracy.py
"""

import re
import json

class AccuracyCalculation:
    def __init__(self, use_llm_judge: bool = True, judge_model: str = "qwen2.5-coder:0.5b", verbose: bool = False):
        self.use_llm_judge = use_llm_judge
        self.judge_model = judge_model
        self.verbose = verbose

    def evaluate_accuracy(self, expected: str, actual: str, initial_input: str = None, runner = None) -> float:
        """
        Routes the evaluation to either the LLM Judge (for partial credit) 
        or the Deterministic fallback based on initialization.
        """
        if self.use_llm_judge and runner and initial_input:
            return self._llm_judge_accuracy(expected, actual, initial_input, runner)
        else:
            return self._deterministic_accuracy(expected, actual)

    def _llm_judge_accuracy(self, expected: str, actual: str, initial_input: str, runner) -> float:
        judge_prompt = (
            f"You are an impartial grader evaluating an AI's response to a complex prompt.\n"
            f"Task Given to AI: \"\"\"{initial_input}\"\"\"\n"
            f"Expected Ideal Output / Requirements: \"\"\"{expected}\"\"\"\n"
            f"Actual AI Output: \"\"\"{actual}\"\"\"\n\n"
            f"Score the AI's output from 0.0 to 1.0 based on this rubric:\n"
            f"- 0.0: Completely wrong, off-topic, or empty.\n"
            f"- 0.3: Addressed the topic but failed major constraints or logic.\n"
            f"- 0.6: Good logic or partial answer, but missed a constraint (e.g., formatting).\n"
            f"- 0.8: Mostly correct, minor hallucination or slightly verbose.\n"
            f"- 1.0: Perfect. Followed all constraints and logic flawlessly.\n\n"
            f"CRITICAL: Output ONLY the float number (e.g., 0.6). Do not output any text, explanations, or formatting."
        )
        
        try:
            judge_response, _ = runner.run_ollama_model(self.judge_model, judge_prompt)
            # Strip out any markdown or conversational text the judge might accidentally output
            clean_score = ''.join(c for c in judge_response if c.isdigit() or c == '.')
            
            # Handle edge cases where multiple decimals are generated (e.g., "0.6.")
            if clean_score.count('.') > 1:
                clean_score = clean_score[:clean_score.find('.', clean_score.find('.') + 1)]
                
            accuracy_score = float(clean_score) if clean_score else 0.0
            return max(0.0, min(1.0, accuracy_score))
        except Exception as e:
            if self.verbose:
                print(f"  [Judge Error] Failed to parse score: {e}")
            return 0.0

    def _deterministic_accuracy(self, expected: str, actual: str) -> float:
        """Original exact-match and regex fallback logic."""
        expected_str = str(expected).strip().lower()
        actual_str = str(actual).strip().lower()

        if expected_str == actual_str:
            return 1.0
        if expected_str in actual_str:
            return 1.0

        expected_numbers = self._extract_numbers(expected_str)
        actual_numbers = self._extract_numbers(actual_str)
        if expected_numbers and expected_numbers == actual_numbers:
            return 1.0

        expected_json = self._extract_json(expected_str)
        actual_json = self._extract_json(actual_str)
        if expected_json and expected_json == actual_json:
            return 1.0

        return 0.0

    def _extract_numbers(self, text: str) -> list[float]:
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(n) for n in numbers]

    def _extract_json(self, text: str) -> dict | None:
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
        return None