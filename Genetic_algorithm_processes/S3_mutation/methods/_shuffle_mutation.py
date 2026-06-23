"""
Genetic_algorithm_processes/S3_mutation/methods/shuffle_mutation.py
"""

# Input a list of objects
# Pick N random points on each parent chromosome to split the list into segments
# Shuffling constant: degree of shuffling applied to the segments 0.0 - 1.0
# Number of moved segments determined by shuffling constant = 1 + int(shuffling constant * (N - 1))
# Offspring: reorder the segments randomly to create the offspring

# Input parameters:
# - N: Number of mutation points (changes size of segments)               | default is random
# - N distribution: Function to determine the mutation points             | default is uniform distribution
# - shuffling constant: Degree of shuffling applied to the segments       | default is full shuffling (1.0)

# Output:
# - offspring: New chromosome created by alternating segments from both parents



import random

class ShuffleMutation:
    def __init__(self, N_segment_cut = None,
                 N_distribution: callable = lambda length: random.randint(1, max(1, length - 1)),
                 shuffling_constant: float = 1.0):
        self.N_segment_cut = N_segment_cut
        self.N_distribution = N_distribution
        self.shuffling_constant = shuffling_constant

    def mutate(self, chromosome):
        length = len(chromosome)
        
        # Determine N (number of cuts) for this specific list
        if self.N_segment_cut is None:
            n_points = random.randint(1, max(1, length - 1))
        else:
            n_points = min(self.N_segment_cut, length - 1)  # Can't have more splits than length-1

        if n_points <= 0 or length <= 1:
            return chromosome.copy()

        # Generate unique sorted points
        points = set()
        attempts = 0
        max_attempts = n_points * 10  # Safety break
        
        while len(points) < n_points and attempts < max_attempts:
            p = self.N_distribution(length)
            if 0 < p < length:
                points.add(p)
            attempts += 1
        
        # If we couldn't generate enough unique points, just use what we have
        crossover_points = sorted(list(points))

        # Split the list into segments
        segments = []
        prev_point = 0
        for point in crossover_points:
            segments.append(chromosome[prev_point:point])
            prev_point = point
        segments.append(chromosome[prev_point:])

        num_segments = len(segments)
        
        # Calculate maximum distance segments can move based on shuffling constant
        # shuffling_constant = 0.0 means no movement
        # shuffling_constant = 1.0 means can move anywhere
        max_distance = max(1, int(self.shuffling_constant * (num_segments - 1)))
        
        # Create new positions for each segment within the allowed distance
        new_positions = list(range(num_segments))
        
        for i in range(num_segments):
            # Calculate allowed range for this segment
            min_pos = max(0, i - max_distance)
            max_pos = min(num_segments - 1, i + max_distance)
            
            # Pick a random position within the allowed range
            new_pos = random.randint(min_pos, max_pos)
            
            # Swap with whatever is currently at that position
            new_positions[i], new_positions[new_pos] = new_positions[new_pos], new_positions[i]
        
        # Reorder segments according to new positions
        shuffled_segments = [segments[new_positions[i]] for i in range(num_segments)]

        # Flatten the list of segments back into a single list
        offspring = [item for segment in shuffled_segments for item in segment]
        
        return offspring


if __name__ == "__main__":
    prompt_chain = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    print("With N=5, shuffling_constant=0.5 (moderate distance):")
    shuffle_mutation_instance = ShuffleMutation(N_segment_cut=2, shuffling_constant=0.5)
    for i in range(5):
        offspring = shuffle_mutation_instance.mutate(prompt_chain)
        print(f"Run {i+1}: {offspring}")
    
    print("\nWith N=5, shuffling_constant=0.2 (low distance):")
    shuffle_low = ShuffleMutation(N_segment_cut=3, shuffling_constant=0.2)
    for i in range(5):
        offspring = shuffle_low.mutate(prompt_chain)
        print(f"Run {i+1}: {offspring}")
    
    print("\nWith N=5, shuffling_constant=1.0 (full shuffling):")
    shuffle_all = ShuffleMutation(N_segment_cut=3, shuffling_constant=1.0)
    for i in range(5):
        offspring = shuffle_all.mutate(prompt_chain)
        print(f"Run {i+1}: {offspring}")