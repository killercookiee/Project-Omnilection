# Input: list of parent prompt chains, one offspring prompt chain
# Output: the offspring prompt chain with each prompt's segments recombined
#         from all parent prompts that share the same model

# For each position in the offspring chain:
#   1. Find all prompts across all parents whose model matches the offspring prompt's model
#   2. If fewer than 2 matches are found, skip (nothing to recombine with)
#   3. Otherwise, run crossover on the offspring prompt's segments using the matching
#      parent prompts as donors, and replace the offspring prompt with the result


"""
Genetic_algorithm_processes/S2_recombination/methods/crossover/model_based_crossover.py
"""
from ._N_crossover_multiparent import NCrossoverMultiparent


class ModelBasedCrossover:
    def __init__(self,
        crossover_method=NCrossoverMultiparent(),
        verbose: bool = False
    ):
        self.crossover_method = crossover_method
        self.verbose = verbose

        if self.verbose:
            print(f"[ModelBasedCrossover] Initialized — method: {type(self.crossover_method).__name__}")

    def select_prompts_by_model(self, parents, model):
        selected_prompts = []
        for parent in parents:
            for prompt in parent:
                if prompt[0] == model:
                    selected_prompts.append(prompt)
        return selected_prompts

    def crossover(self, parents, offspring):
        SHADES = ['\033[47m', '\033[100m', '\033[107m', '\033[40m', '\033[43m', '\033[46m']
        RESET = '\033[0m'

        def prompt_block(parent_idx, text):
            shade = SHADES[parent_idx % len(SHADES)]
            label = f" {text[:10]}{'…' if len(text) > 10 else ''} "
            return f"{shade}{label}{RESET}"

        if self.verbose:
            print(f"\n[ModelBasedCrossover] Crossing {len(parents)} parents | Offspring has {len(offspring)} prompts")

        for i in range(len(offspring)):
            offspring_prompt = offspring[i]
            model = offspring_prompt[0]
            matching_parent_prompts = self.select_prompts_by_model(parents, model)

            if self.verbose:
                print(f"\n{'─'*40}")
                print(f"  Prompt [{i+1}/{len(offspring)}]  model: {model}")
                print(f"  Offspring : {prompt_block(0, ''.join(offspring_prompt[1:]))}")

                parent_row = ""
                for p_idx, prompt in enumerate(matching_parent_prompts):
                    text = "".join(prompt[1:])
                    parent_row += prompt_block(p_idx, text) + " "
                print(f"  Donors  : {parent_row}")

            if len(matching_parent_prompts) < 2:
                if self.verbose:
                    print(f"  ⚠️  Only {len(matching_parent_prompts)} donor(s) found for model '{model}' — skipping crossover")
                continue

            # Recombine the offspring prompt's segments using the matching parent prompts as donors
            new_prompt = self.crossover_method.crossover([offspring_prompt] + matching_parent_prompts)
            offspring[i] = new_prompt

            if self.verbose:
                result_text = "".join(new_prompt[1:]) if isinstance(new_prompt, tuple) else str(new_prompt[:3])
                print(f"  Result  : {SHADES[0]} {result_text[:100]}{'…' if len(result_text) > 100 else ''} {RESET}")

        if self.verbose:
            print(f"\n{'─'*40}")
            print(f"  ✅  Crossover complete — {len(offspring)} prompts processed")
            print(f"{'─'*40}\n")

        return offspring


if __name__ == "__main__":
    parents = [
        [("gemma3:270m", "Summarize the task and give a hint of the answer. ", "Make it concise and clear."),
         ("qwen2.5-coder:0.5b", "Provide 1 word answer only")],

        [("smollm:360m", "Summarize the task. ", "Give a hint of the completely wrong answer."),
         ("qwen2:0.5b", "Give a wrong answer. ", "Explain why the right answer is correct and the wrong answer is wrong.")],

        [("qwen2:0.5b", "Try to give a wrong but sort of right answer. ")]
    ]
    offspring = [('gemma3:270m', 'Summarize the task and give a hint of the answer. ', 'Make it concise and clear.'),
                 ('qwen2:0.5b', 'Give a wrong answer. ', 'Explain why the right answer is correct and the wrong answer is wrong.')]

    model_random_crossover = ModelBasedCrossover(verbose=True)
    new_offspring = model_random_crossover.crossover(parents, offspring)
    print(f"Final offspring: {new_offspring}")