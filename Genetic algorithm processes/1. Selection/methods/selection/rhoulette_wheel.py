# Input a dictionary of individuals with their fitness scores
# Output a list of selected individuals based on rhoulette wheel selection

import random

class RouletteWheelSelection:
    def __init__(self, population, num_selections):
        """
        Roulette Wheel Selection method.
        
        Parameters:
        - population: A dictionary where keys are individuals and values are their fitness scores (dict)
        - num_selections: Number of individuals to select (int)
        """
        self.population = population
        self.num_selections = num_selections

    def roulette_wheel_selection(self):
        """
        Roulette Wheel Selection method.
        
        Parameters:
        - population: A dictionary where keys are individuals and values are their fitness scores (dict)
        - num_selections: Number of individuals to select (int)
        
        Returns:
        - selected_individuals: A list of selected individuals (list)
        """
        total_fitness = sum(self.population.values())
        selected_individuals = []
        
        for _ in range(self.num_selections):
            pick = random.uniform(0, total_fitness)
            current = 0.0
            for individual, fitness in self.population.items():
                current += fitness
                if current > pick:
                    selected_individuals.append(individual)
                    break
        
        return selected_individuals