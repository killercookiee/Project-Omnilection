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

class SingleHomologousCrossover:
    def __init__(self, parent_1, parent_2, crossover_point_1_distribution=None, crossover_point_2_distribution=None):
        pass