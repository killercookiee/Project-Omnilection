# Input 2 parent chromosomes (lists of genes)
# Pick N random crossover points on each parent chromosome to split the chromosomes into segments
# Offspring: Zip the segments from Parent 1 and Parent 2 alternately to create the offspring

# Input parameters:
# - parent_1: First parent chromosome
# - parent_2: Second parent chromosome
# - N_1 distribution: Function to determine the crossover points from parent 1       | default is uniform distribution
# - N_2 distribution: Function to determine the crossover points from parent 2       | default is uniform distribution

# Output:
# - offspring: New chromosome created by alternating segments from both parents

class NHomologousCrossover:
    def __init__(self, parent_1, parent_2, N_1_distribution=None, N_2_distribution=None):
        pass