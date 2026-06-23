# Input: a list of the population of individuals (prompt chains)
# Output: a list of pairs (2 or more) of the paired individuals for recombination

# Customizable parameters:
# - size_of_pairs: number of individuals in each pair (default is 2 for pairwise recombination)
# - number_of_pairs: number of pairs to create (default is None, meaning all possible pairs)

# Input: a list of the population of individuals (prompt chains)
# Output: a list of pairs (2 or more) of the paired individuals for recombination

# Customizable parameters:
# - size_of_pairs: number of individuals in each pair (default is 2 for pairwise recombination)
# - number_of_pairs: number of pairs to create (default is None, meaning all possible pairs)

"""
Genetic_algorithm_processes/S2_recombination/methods/pairing/random_pairing.py
"""

import random


class RandomPairing:
    def __init__(self,
        size_of_pairs: int = 2,
        number_of_pairs: int = 1,
        verbose: bool = False
    ):
        self.size_of_pairs = size_of_pairs
        self.number_of_pairs = number_of_pairs
        self.verbose = verbose

        if self.verbose:
            print(f"[RandomPairing] Initialized — pair size: {self.size_of_pairs} | pairs to generate: {self.number_of_pairs}")

    def pair(self, population):
        SHADES = ['\033[47m', '\033[100m', '\033[107m', '\033[40m', '\033[43m', '\033[46m']
        RESET = '\033[0m'

        def individual_block(idx):
            shade = SHADES[idx % len(SHADES)]
            return f"{shade}  P{idx+1}  {RESET}"

        # Sample with replacement: each pair is drawn independently
        paired_individuals = [
            random.choices(population, k=self.size_of_pairs)
            for _ in range(self.number_of_pairs)
        ]

        if self.verbose:
            print(f"\n[RandomPairing] Population: {len(population)} | Pair size: {self.size_of_pairs} | Pairs generated: {len(paired_individuals)}")
            print()
            for pair_idx, pair in enumerate(paired_individuals):
                indices = [population.index(ind) for ind in pair]
                blocks = " + ".join(individual_block(idx) for idx in indices)
                print(f"  Pair {pair_idx+1}: │ {blocks} │")
            print()

        return paired_individuals


if __name__ == "__main__":
    prompt_chain_population = [
        [("gemma3:270m", "Summarize the task and give a hint of the answer. ", "Make it concise and clear."),
         ("qwen2.5-coder:0.5b", "Provide 1 word answer only")],

        [("smollm:360m", "Summarize the task. ", "Give a hint of the completely wrong answer."),
         ("qwen2:0.5b", "Give a wrong answer. ", "Explain why the right answer is correct and the wrong answer is wrong.")],

        [("deepseek-coder:latest", "Try to give a wrong but sort of right answer. ")]
    ]

    random_pairing_instance = RandomPairing(size_of_pairs=2, number_of_pairs=10, verbose=True)
    paired = random_pairing_instance.pair(prompt_chain_population)
    print(f"Final — {len(paired)} pairs generated")