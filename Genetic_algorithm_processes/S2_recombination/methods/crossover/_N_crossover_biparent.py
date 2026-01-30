# Input 2 parent chromosomes (lists of genes)
# Pick N random crossover points on each parent chromosome to split the chromosomes into segments

# Offspring: Zip the segments from Parent 1 and Parent 2 alternately to create the offspring

# Input parameters:
# - parent_1: First parent chromosome
# - parent_2: Second parent chromosome
# - N_1: Number of crossover points on parent 1, determines size of segments      | default is 2
# - segment_size_1: fixed size of segments from parent 1, integer > 0, overrides N_1 if provided                | default is None
# - N_1 distribution: Function to return the crossover point depending on the length of the chromosome      | default is uniform distribution

# Output:
# - offspring: One chromosome created by alternating segments from both parents


import random

class NCrossoverBiparent:
    def __init__(self, parent_1, parent_2, crossover_num=None, segment_size=None, distribution=None):
        self.parent_1 = parent_1
        self.parent_2 = parent_2
        self.crossover_num = crossover_num if crossover_num is not None else random.randint(1, max(1, min(len(parent_1), len(parent_2)) - 1))
        # number of segments = crossover_num + 1
        self.segment_size = segment_size
        
        # Default uniform distribution: returns a random index between 1 and len(chromosome)-1
        self.random_uniform_distribution = lambda length: random.randint(1, max(1, length - 1))
        self.distribution = distribution or self.random_uniform_distribution
    
    def _get_crossover_points(self, chromosome_length):
        """
        Calculate crossover points. 
        If segment_size is set, N is calculated dynamically based on parent length.
        Otherwise, it uses the fixed N (crossover_num).
        """
        # Determine N (number of cuts) for this specific parent
        if self.segment_size:
            # e.g. Length 10, size 2 => 5 segments => 4 cuts
            n_points = int(chromosome_length / self.segment_size) - 1
        else:
            n_points = self.crossover_num

        if n_points <= 0:
            return []

        # Generate unique sorted points
        points = set()
        attempts = 0
        max_attempts = n_points * 5 # Safety break
        
        while len(points) < n_points and attempts < max_attempts:
            p = self.distribution(chromosome_length)
            if 0 < p < chromosome_length:
                points.add(p)
            attempts += 1
            
        return sorted(list(points))

    def _split_chromosome(self, chromosome, crossover_points):
        """Splits the chromosome list at the specified indices."""
        if not crossover_points:
            return [chromosome]
        
        segments = []
        prev_point = 0
        
        for point in crossover_points:
            segments.append(chromosome[prev_point:point])
            prev_point = point
        
        # Add the tail (last segment)
        segments.append(chromosome[prev_point:])
        
        return segments
    
    def _zip_segments_alternating(self, segments_1, segments_2):
        """
        Creates ONE offspring by alternating segments.
        Order: P1_Seg1, P2_Seg2, P1_Seg3, P2_Seg4...
        """
        offspring = []
        
        # We iterate as far as the longest parent allows
        max_segments = max(len(segments_1), len(segments_2))
        
        for i in range(max_segments):
            # EVEN index (0, 2, 4...) -> Take from PARENT 1
            if i % 2 == 0:
                if i < len(segments_1):
                    offspring.extend(segments_1[i])
            
            # ODD index (1, 3, 5...) -> Take from PARENT 2
            else:
                if i < len(segments_2):
                    offspring.extend(segments_2[i])
        
        return offspring
    
    def crossover(self):
        # 1. Get points (Calculated independently for non-homologous lengths)
        points_1 = self._get_crossover_points(len(self.parent_1))
        points_2 = self._get_crossover_points(len(self.parent_2))
        
        # 2. Split parents into segments
        segments_1 = self._split_chromosome(self.parent_1, points_1)
        segments_2 = self._split_chromosome(self.parent_2, points_2)
        
        # 3. Zip segments (Head from P1, alternate, single offspring)
        offspring = self._zip_segments_alternating(segments_1, segments_2)
        
        return offspring


if __name__ == "__main__":
    # Parent 1 (Longer)
    parent_1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
    # Parent 2 (Shorter, distinct values for clarity)
    parent_2 = ['a', 'b', 'c', 'd', 'e']
    
    # Setup crossover (2 cuts = 3 segments approx)
    crossover = NCrossoverBiparent(parent_1, parent_2, crossover_num=2)
    
    # Run
    offspring = crossover.crossover()
    
    print(f"Parent 1: {parent_1}")
    print(f"Parent 2: {parent_2}")
    print("-" * 40)
    print(f"Offspring: {offspring}")

    # To visualize the segments logic explicitly:
    # If P1 segments were: [1,1,1], [1,1,1], [1,1,1,1]
    # If P2 segments were: [a], [b,c], [d,e]
    # Offspring would be:  [1,1,1] + [b,c] + [1,1,1,1]