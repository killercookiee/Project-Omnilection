"""
Genetic_algorithm_processes/S5_simulated_annealing/methods/micro_mutation.py
"""

import random
from Genetic_algorithm_processes.ollama_run import PromptChainRunner

class MicroMutation:
    def __init__(self, runner: PromptChainRunner, mutator_model: str = "qwen2.5-coder:0.5b", verbose: bool = False):
        self.runner = runner
        self.mutator_model = mutator_model
        self.verbose = verbose

    def mutate(self, chain: list, current_temp: float, initial_temp: float) -> list:
        # Genetic Armor: Ensure chain is actually a list
        if not chain or not isinstance(chain, list):
            return chain

        # Calculate "Heat" (1.0 = Hot/More changes, 0.1 = Cold/Tiny changes)
        heat_ratio = max(0.1, min(1.0, current_temp / initial_temp))
        
        # Pick a random step in the chain
        step_idx = random.randint(0, len(chain) - 1)
        step = chain[step_idx]
        
        # ── Genetic Armor: Protect against malformed tuples ──
        if not isinstance(step, (list, tuple)) or len(step) < 2:
            return chain # Safely abort mutation on broken genetic material
            
        model_name = step[0]
        segments = step[1]
        
        # ── Genetic Armor: Ensure segments is a list of strings ──
        if isinstance(segments, str):
            segments = [segments]
        elif not isinstance(segments, list) or len(segments) == 0:
            return chain

        # Pick a random segment within the step
        seg_idx = random.randint(0, len(segments) - 1)
        target_segment = str(segments[seg_idx])

        if len(target_segment.strip()) < 5:
            return chain 

        # Temperature-Aware Prompting
        if heat_ratio > 0.5:
            instruction = "Swap 1 or 2 words with synonyms to improve flow. Do NOT rewrite the sentence."
        else:
            instruction = "Make a microscopic adjustment. Change a single punctuation mark, capitalization, or remove a single redundant word."

        mutator_prompt = (
            f"You are a strict copyeditor. Your task is to apply a MINOR tweak to the following text.\n"
            f"Constraint: {instruction}\n"
            f"CRITICAL: Output ONLY the edited text. No explanations. Preserve the original meaning exactly.\n\n"
            f"Text: \"\"\"{target_segment}\"\"\"\n"
            f"Edited Text:\n"
        )

        output, _ = self.runner.run_ollama_model(self.mutator_model, mutator_prompt)
        mutated_text = output.strip().strip('"').strip("'")
        
        # If the LLM failed to return a valid mutation, abort
        if not mutated_text or len(mutated_text) < 2 or mutated_text == target_segment:
            return chain

        # Reconstruct the chain with the new genetic material
        new_segments = list(segments)
        new_segments[seg_idx] = mutated_text
        
        new_chain = list(chain)
        new_chain[step_idx] = (model_name, new_segments)

        if self.verbose:
            print(f"  [MicroMutation] Temp {heat_ratio:.1f}x -> Tweak applied.")

        return new_chain