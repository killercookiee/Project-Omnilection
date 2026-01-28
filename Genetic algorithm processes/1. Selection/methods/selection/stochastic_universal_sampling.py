# Input a dictionary of individuals with their fitness scores
# Output a list of selected individuals based on stochastic universal sampling


import random

class StochasticUniversalSampling:
    def __init__(self, selection_ratio=0.5):
        self.selection_ratio = selection_ratio

    def stochastic_universal_sampling(self, population):
        """
        Stochastic Universal Sampling selection method.
        
        Parameters:
        - population: A dictionary where keys are individuals and values are their fitness scores (dict)
        - num_selections: Number of individuals to select (int)
        
        Returns:
        - selected_individuals: A list of selected individuals (list)
        """
        num_selections = max(1, int(len(population) * self.selection_ratio))

        total_fitness = sum(population.values())
        pointer_distance = total_fitness / num_selections
        start_point = random.uniform(0, pointer_distance)
        
        pointers = [start_point + i * pointer_distance for i in range(num_selections)]
        
        selected_individuals = []
        cumulative_fitness = 0.0
        individual_iter = iter(population.items())
        current_individual, current_fitness = next(individual_iter)
        
        for pointer in pointers:
            while cumulative_fitness + current_fitness < pointer:
                cumulative_fitness += current_fitness
                current_individual, current_fitness = next(individual_iter)
            selected_individuals.append(current_individual)
        
        return selected_individuals
    

if __name__ == "__main__":
    sus_method = StochasticUniversalSampling(selection_ratio=0.5)

    population = {
        'individual_1': 10,
        'individual_2': 30,
        'individual_3': 20,
        'individual_4': 40
    }
    num_selections = 3

    selected = sus_method.stochastic_universal_sampling(population, num_selections)
    print(f"Selected Individuals: {selected}")