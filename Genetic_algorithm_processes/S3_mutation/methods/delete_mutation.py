"""
Genetic_algorithm_processes/S3_mutation/methods/delete_mutation.py
"""

# Input a list of objects
# Pick a random segment and delete it from the list
# Offspring: New list with the segment removed

# Customization parameters:
# - min_segment_size: Minimum size of the segment to delete | default is 1
# - max_segment_size: Maximum size of the segment to delete | default is half the length of the list


import random

class DeleteMutation:
    """
    DeleteMutation removes a random segment from a chromosome.

    Parameters:
    - min_segment_fraction: Minimum fraction of chromosome length to delete (0 to 1)
    - max_segment_fraction: Maximum fraction of chromosome length to delete (0 to 1)
    """
    def __init__(self,
        min_segment_fraction: float =   0.0,
        max_segment_fraction: float =   0.5
    ):
        self.min_segment_fraction = min_segment_fraction
        self.max_segment_fraction = max_segment_fraction

    def mutate(self, chromosome):
        length = len(chromosome)
        if length == 0:
            return chromosome.copy()

        # Compute segment sizes relative to chromosome length
        min_size = max(1, int(length * self.min_segment_fraction))
        max_size = max(1, int(length * self.max_segment_fraction))

        # Clamp max_size to chromosome length
        max_size = min(max_size, length)

        # Choose a random segment size
        segment_size = random.randint(min_size, max_size)

        # Choose random start index
        start_index = random.randint(0, length - segment_size)
        end_index = start_index + segment_size

        # Create new chromosome with segment removed
        new_chromosome = chromosome[:start_index] + chromosome[end_index:]
        return new_chromosome

    

if __name__ == "__main__":
    prompt_chain = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    print("Delete Mutation with min_segment_fraction=0.1, max_segment_fraction=0.3:")
    delete_mutation_instance = DeleteMutation(min_segment_fraction=0.1, max_segment_fraction=0.3)
    for i in range(5):
        offspring = delete_mutation_instance.mutate(prompt_chain)
        print(f"Run {i+1}: {offspring}")