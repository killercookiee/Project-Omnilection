"""
Genetic_algorithm_processes/S3_mutation/methods/semantic_llm_mutation.py
"""

import random
from Genetic_algorithm_processes.ollama_run import PromptChainRunner

class SemanticLLMMutation:
    def __init__(self, runner: PromptChainRunner, mutator_model: str = "qwen2.5-coder:0.5b", verbose: bool = False):
        self.runner = runner
        self.mutator_model = mutator_model
        self.verbose = verbose
        
        # The distinct evolutionary paths we want to force the prompt to take
        self.modes = [
            "EXPAND: Make this prompt highly detailed, explicit, and comprehensive.",
            "SUMMARIZE: Condense this prompt to be extremely concise and direct.",
            "PARAPHRASE: Rewrite this using advanced vocabulary and a completely different sentence structure.",
            "TRANSLATE-THINKING: Rewrite the prompt by translating its core logic to another language and back, resulting in a unique phrasing."
        ]

    def mutate(self, chain: list[tuple]) -> list[tuple]:
        if not chain:
            return chain

        # 1. Pick a random step in the chain, and a random text segment in that step
        step_idx = random.randint(0, len(chain) - 1)
        model_name, segments = chain[step_idx]
        
        if not segments:
            return chain

        seg_idx = random.randint(0, len(segments) - 1)
        target_segment = segments[seg_idx]

        if len(target_segment.strip()) < 5:
            return chain # Too short to meaningfully mutate (likely just a space or punctuation)

        # 2. Pick a semantic mutation mode
        mode = random.choice(self.modes)

        # 3. Construct the strict wrapper prompt for the Mutator LLM
        mutator_prompt = (
            f"You are a prompt engineering assistant. Your task is to rewrite the following prompt segment according to this stylistic instruction: '{mode}'.\n\n"
            f"CRITICAL CONSTRAINTS:\n"
            f"1. ONLY output the rewritten prompt text. No pleasantries, no explanations.\n"
            f"2. DO NOT answer the prompt yourself.\n"
            f"3. You MUST preserve the original core task, input data references, and expected output formatting role.\n\n"
            f"Original Segment:\n\"\"\"{target_segment}\"\"\"\n\n"
            f"Rewritten Segment:\n"
        )

        if self.verbose:
            print(f"  [SemanticMutation] Prompting {self.mutator_model} to {mode.split(':')[0]}...")

        # 4. Execute the LLM call to mutate the text
        output, metrics = self.runner.run_ollama_model(self.mutator_model, mutator_prompt)
        
        # Clean the output of any stray quotes the LLM might have added
        mutated_text = output.strip().strip('"').strip("'")
        
        # Fallback: If the mutator failed or returned empty, return original
        if not mutated_text or len(mutated_text) < 2:
            return chain

        # 5. Reconstruct the chain with the new genetic material
        new_segments = list(segments)
        new_segments[seg_idx] = mutated_text
        
        new_chain = list(chain)
        new_chain[step_idx] = (model_name, new_segments)

        return new_chain