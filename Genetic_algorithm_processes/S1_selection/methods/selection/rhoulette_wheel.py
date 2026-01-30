# Input a dictionary of individuals with their fitness scores
# Output a list of selected individuals based on rhoulette wheel selection

import random
class RouletteWheelSelection:
    def __init__(self, selection_ratio=None):
        """
        Roulette Wheel Selection method.
        
        Parameters:
        - population: A dictionary where keys are individuals and values are their fitness scores (dict)
        - num_selections: Number of individuals to select (int)
        """
        self.selection_ratio = selection_ratio if selection_ratio is not None else random.uniform(0.3, 0.7)

    def roulette_wheel_selection(self, population):
        """
        Roulette Wheel Selection method.
        
        Parameters:
        - population: A dictionary where keys are individuals and values are their fitness scores (dict)
        - num_selections: Number of individuals to select (int)
        
        Returns:
        - selected_individuals: A list of selected individuals (list)
        """
        num_selections = max(1, int(len(population) * self.selection_ratio))

        total_fitness = sum(population.values())
        selected_individuals = []
        
        for _ in range(num_selections):
            pick = random.uniform(0, total_fitness)
            current = 0.0
            for individual, fitness in population.items():
                current += fitness
                if current > pick:
                    selected_individuals.append(individual)
                    break
        
        return selected_individuals
    

if __name__ == "__main__":
    selector = RouletteWheelSelection(selection_ratio=0.5)

    # Example population with fitness scores
    population = {
        'individual_1': 10,
        'individual_2': 30,
        'individual_3': 20,
        'individual_4': 40
    }

    selected = selector.roulette_wheel_selection(population)
    print("Selected Individuals:", selected)