# Input: a list of the population of individuals (prompt chains)
# Output: a list of pairs (2 or more) of the paired individuals for recombination

# Customizable parameters:
# - size_of_pairs: number of individuals in each pair (default is 2 for pairwise recombination)
# - number_of_pairs: number of pairs to create (default is None, meaning all possible pairs)


class RandomPairing:
    def __init__(self, size_of_pairs=2, number_of_pairs=None):
        self.size_of_pairs = size_of_pairs
        self.number_of_pairs = number_of_pairs if number_of_pairs is not None else 1

    def random_pairing(self, population):
        import random
        shuffled_population = population[:]
        random.shuffle(shuffled_population)
        
        paired_individuals = []
        for i in range(0, len(shuffled_population), self.size_of_pairs):
            pair = shuffled_population[i:i + self.size_of_pairs]
            if len(pair) == self.size_of_pairs:
                paired_individuals.append(pair)
                # Stop if we've reached the desired number of pairs
                if self.number_of_pairs and len(paired_individuals) >= self.number_of_pairs:
                    break
        
        return paired_individuals
    

if __name__ == "__main__":
    prompt_chain_population = [
        [("gpt-3.5-turbo", "Prompt", "1"), ("gpt-4", "Prompt", "2")],
        [("gpt-4", "Prompt 3"), ("gpt-3.5-turbo", "Prompt 4")],
        [("gpt-3.5-turbo", "Prompt", "5"), ("gpt-4", "Prompt", "6")],
        [("gpt-4", "Prompt", "7"), ("gpt-3.5-turbo", "Prompt", "8")]
    ]
    
    random_pairing_instance = RandomPairing(size_of_pairs=2)
    paired = random_pairing_instance.random_pairing(prompt_chain_population)
    print(f"Paired Individuals: {paired}")

    # Paired Individuals: [[[{'gpt-3.5-turbo', 'Prompt 1'}, {'Prompt 2', 'gpt-4'}], [{'Prompt 3', 'gpt-4'}, {'gpt-3.5-turbo', 'Prompt 4'}]]]