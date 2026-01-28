# Input 1 parent chromosome (lists of genes)
# Pick N random points on each parent chromosome to split the chromosomes into segments
# Shuffling constant: degree of shuffling applied to the segments 0.0 - 1.0
# Number of moved segments determined by shuffling constant = 1 + int(shuffling constant * (N - 1))
# Offspring: reorder the segments randomly to create the offspring

# Input parameters:
# - parent: Parent chromosome
# - N: Number of mutation points
# - N distribution: Function to determine the mutation points             | default is uniform distribution
# - shuffling constant: Degree of shuffling applied to the segments       | default is full shuffling (1.0)

# Output:
# - offspring: New chromosome created by alternating segments from both parents

class ShuffleMutation:
    def __init__(self, chromosome, N=None, N_distribution=None, shuffling_constant=1.0):
        pass