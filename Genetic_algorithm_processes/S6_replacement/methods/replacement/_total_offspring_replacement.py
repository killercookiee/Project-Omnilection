# Class input total population cap
# Function input offspring population and current population fitness
# function replaces lowest population fitness members with offsprings completely


class TotalOffspringReplacement:
    def __init__(self):
        pass

    def replace(self, offspring_population: list[list[tuple]], current_population_fitness: list[tuple], population_cap: int) -> list[tuple]:
        """
        lowest fitness individuals in current population fitness with offspring fitness to fill up until population cap without checking the offspring fitness

        offspring_population: list of new individuals (each individual is a of prompt chain, no fitness score attached)
        current_population_fitness: list of current individuals with their fitness scores (each is a tuple (individual, fitness_score))
        returns: list of individuals after replacement
        """
        # lowest fitness individuals in current population fitness with offspring fitness to fill up until population cap without checking the offspring fitness
        combined_population_fitness = current_population_fitness + [(offspring, None) for offspring in offspring_population]
        combined_population_fitness = combined_population_fitness[:population_cap]
        return combined_population_fitness
    

if __name__ == "__main__":
    offspring_population = [
        [("gpt-3.5-turbo", "This is an ", "input prompt"), ("gpt-4", "This ", "another ", "input ", "prompt")],
        [("gpt-4", "Different input prompt here"), ("gpt-3.5-turbo", "", "Yet another prompt input")],
        [("gpt-3.5-turbo", "Sample prompt one"), ("gpt-4", "Sample prompt two"), ("gpt-3.5-turbo", "Sample prompt three")],
        [("gpt-4", "First prompt segment"), ("gpt-4", "Second prompt segment")],
        [("gpt-3.5-turbo", "Only one prompt in this chain")]
    ] 
    current_population_fitness = [
        ([("gpt-3.5-turbo", "Old prompt chain 1")], 0.8),
        ([("gpt-4", "Old prompt chain 2")], 0.6),
        ([("gpt-3.5-turbo", "Old prompt chain 3")], 0.9)
    ]
    population_cap = 4
    replacement_instance = TotalOffspringReplacement(population_cap=population_cap)
    new_population_fitness = replacement_instance.replace(offspring_population, current_population_fitness)


    print(f"New Population fitness after Replacement: {new_population_fitness}")