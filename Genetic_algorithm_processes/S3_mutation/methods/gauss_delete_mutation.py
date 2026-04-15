"""
Genetic_algorithm_processes/S3_mutation/methods/gauss_delete_mutation.py
"""

# Input a list of objects
# Pick a random segment and delete it from the list based on normal distribution
# Offspring: New list with the segment removed

# Customization parameters:
# - position (mean): Center position for deletion (0 to 1, where 0 is start, 1 is end) | default is uniform random
# - avg_segment_size: Average size of segment to delete as fraction of chromosome length (0 to 1) | default is uniform random between 0.01 and 0.3
# - std_dev: Standard deviation for normal distribution (alternative to avg_segment_size) | calculated from avg_segment_size if not provided


class GaussDeleteMutation:
    def __init__(self,
        position: float = 0.5,
        avg_segment_size: float = 0.2,
        std_dev: float = 0.12533
    ):
        self.position = position  # mean μ (0 to 1)
        self.avg_segment_size = avg_segment_size  # average segment size (0 to 1)
        self.std_dev = std_dev  # standard deviation (0 to 1)
        
        # If both avg_segment_size and std_dev are provided, std_dev takes precedence
        if self.std_dev is None and self.avg_segment_size is not None:
            # Convert avg_segment_size to std_dev: σ = avg_segment_size * sqrt(π/8)
            import math
            self.std_dev = self.avg_segment_size * math.sqrt(math.pi / 8)

    def mutate(self, chromosome):
        import random
        import math
        
        length = len(chromosome)
        if length == 0:
            return chromosome.copy()

        # Determine standard deviation (σ)
        if self.std_dev is None:
            # Default: random avg_segment_size between 0.1 and 0.4
            default_avg_size = random.uniform(0.1, 0.4)
            std_dev = default_avg_size * math.sqrt(math.pi / 8)
        else:
            std_dev = self.std_dev

        # Determine mean position (μ)
        if self.position is None:
            mean = random.gauss(0.5, std_dev)
        else:
            mean = self.position
        
        # Sample X from normal distribution N(μ, σ)
        X = random.gauss(mean, std_dev)
        
        # Calculate mirror position: 2μ - X
        mirror = 2 * mean - X
        
        # Clamp both positions to [0, 1]
        X = max(0, min(1, X))
        mirror = max(0, min(1, mirror))
        
        # Convert to actual indices
        start_pos = min(X, mirror)
        end_pos = max(X, mirror)
        
        start_index = int(start_pos * length)
        end_index = int(end_pos * length)
        
        # Ensure end_index doesn't exceed length
        end_index = min(end_index, length)
        
        # Create new chromosome with the segment removed
        new_chromosome = chromosome[:start_index] + chromosome[end_index:]
        return new_chromosome
    

if __name__ == "__main__":
    prompt_chain = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    print("Standard Delete Mutation with position=0.5, avg_segment_size=0.2:")
    delete_mutation_instance = GaussDeleteMutation(position=0.5, avg_segment_size=0.2)
    for i in range(5):
        offspring = delete_mutation_instance.mutate(prompt_chain)
        print(f"Run {i+1}: {offspring}")
    
    print("\nStandard Delete Mutation with position=0.3, avg_segment_size=1:")
    delete_mutation_instance2 = GaussDeleteMutation(position=0.3, avg_segment_size=0.8)
    for i in range(5):
        offspring = delete_mutation_instance2.mutate(prompt_chain)
        print(f"Run {i+1}: {offspring}")
    
    print("\nStandard Delete Mutation with default parameters (random position and size):")
    delete_mutation_instance3 = GaussDeleteMutation()
    for i in range(5):
        offspring = delete_mutation_instance3.mutate(prompt_chain)
        print(f"Run {i+1}: {offspring}")