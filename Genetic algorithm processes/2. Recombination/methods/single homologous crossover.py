# Input 2 parent chromosomes (lists of genes)
# Pick a random crossover point on each parent chromosome
# Head: genes before the crossover point
# Tail: genes from the crossover point to the end
# Offspring: Head of Parent 1 + Tail of Parent 2

# Input parameters:
# - parent_1: First parent chromosome (head)
# - parent_2: Second parent chromosome (tail)
# - crossover_point_1 distribution: Function to determine the crossover point       | default is uniform distribution
# - crossover_point_2 distribution: Function to determine the crossover point       | default is uniform distribution

# Output:
# - offspring: New chromosome created by combining head of Parent 1 and tail of Parent 2

import random

class SingleHomologousCrossover:
    def __init__(self, parent_1, parent_2, crossover_point_1_distribution=None, crossover_point_2_distribution=None):
        """
        Initialize Single Homologous Crossover
        
        Parameters:
        - parent_1: First parent chromosome (list of genes) - provides head
        - parent_2: Second parent chromosome (list of genes) - provides tail
        - crossover_point_1_distribution: Function(length) that returns crossover point for parent 1
        - crossover_point_2_distribution: Function(length) that returns crossover point for parent 2
        """
        self.parent_1 = parent_1
        self.parent_2 = parent_2
        
        # Default uniform distribution: returns random point between 1 and length-1
        if crossover_point_1_distribution is None:
            self.crossover_point_1_distribution = lambda length: random.randint(1, max(1, length - 1)) if length > 1 else length
        else:
            self.crossover_point_1_distribution = crossover_point_1_distribution
            
        if crossover_point_2_distribution is None:
            self.crossover_point_2_distribution = lambda length: random.randint(1, max(1, length - 1)) if length > 1 else length
        else:
            self.crossover_point_2_distribution = crossover_point_2_distribution
    
    def crossover(self):
        """
        Perform single homologous crossover
        
        Returns:
        - offspring: New chromosome (Head of Parent 1 + Tail of Parent 2)
        """
        # Get crossover points from distributions
        crossover_point_1 = self.crossover_point_1_distribution(len(self.parent_1))
        crossover_point_2 = self.crossover_point_2_distribution(len(self.parent_2))
        
        # Validate crossover points
        crossover_point_1 = max(0, min(crossover_point_1, len(self.parent_1)))
        crossover_point_2 = max(0, min(crossover_point_2, len(self.parent_2)))
        
        # Head of Parent 1 (genes before crossover point)
        head_1 = self.parent_1[:crossover_point_1]
        
        # Tail of Parent 2 (genes from crossover point to end)
        tail_2 = self.parent_2[crossover_point_2:]
        
        # Combine to create offspring
        offspring = head_1 + tail_2
        
        return offspring


# Example usage:
if __name__ == "__main__":
    # Example parents
    parent_1 = [1, 2, 3, 4, 5, 6, 7, 8]
    parent_2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    
    # Create crossover instance with default uniform distribution
    crossover = SingleHomologousCrossover(parent_1, parent_2)
    
    # Perform crossover
    offspring = crossover.crossover()
    
    print("Parent 1:", parent_1)
    print("Parent 2:", parent_2)
    print("Offspring:", offspring)
    
    # Example with custom distribution
    def crossover_at_midpoint(length):
        return length // 2
    
    def crossover_at_quarter(length):
        return length // 4
    
    crossover_custom = SingleHomologousCrossover(
        parent_1, 
        parent_2, 
        crossover_point_1_distribution=crossover_at_midpoint,
        crossover_point_2_distribution=crossover_at_quarter
    )
    
    offspring_custom = crossover_custom.crossover()
    print("\nWith custom distribution (mid and quarter):")
    print("Offspring:", offspring_custom)
    
    # Multiple runs to see variation
    print("\nMultiple runs with default distribution:")
    for i in range(5):
        offspring = crossover.crossover()
        print(f"Run {i+1}: {offspring}")