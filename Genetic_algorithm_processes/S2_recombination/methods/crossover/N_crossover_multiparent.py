# Input N parent chromosomes (lists of genes)
# Pick N random crossover points on each parent chromosome to split the chromosomes into segments

# Offspring: Rotate through segments from all parents to create the offspring
# Order: P1_Seg1, P2_Seg2, P3_Seg3, ..., P1_Seg(N+1), P2_Seg(N+2), etc.

# Input parameters:
# - parents: List of parent chromosomes
# - crossover_num: Number of crossover points per parent, determines size of segments | default is random
# - segment_size: fixed size of segments, integer > 0, overrides crossover_num if provided | default is None
# - distribution: Function to return the crossover point depending on the length of the chromosome | default is uniform distribution

# Output:
# - offspring: One chromosome created by rotating segments from all parents


import random

class NCrossoverMultiparent:
    def __init__(self, crossover_num=None, segment_size=None, distribution=None, number_offsprings=1):
        self.crossover_num = crossover_num
        self.segment_size = segment_size
        self.number_offsprings = number_offsprings
        
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
        max_attempts = n_points * 5  # Safety break
        
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
    
    def _rotate_segments_multiparent(self, all_segments, parent_num):
        """
        Creates ONE offspring by rotating through segments from all parents.
        Order: P1_Seg1, P2_Seg2, P3_Seg3, ..., P1_Seg(parent_num+1), etc.
        
        Args:
            all_segments: List of segment lists, one per parent
            parent_num: Number of parents
        """
        offspring = []
        
        # Find the maximum number of segments among all parents
        max_segments = max(len(segments) for segments in all_segments)
        
        # Iterate through all possible segment positions
        for seg_idx in range(max_segments):
            # Determine which parent to take from (rotate through parents)
            parent_idx = seg_idx % parent_num
            
            # If this parent has a segment at this index, add it
            if seg_idx < len(all_segments[parent_idx]):
                offspring.extend(all_segments[parent_idx][seg_idx])
        
        return offspring
    
    def crossover(self, parents):
        """
        Perform crossover operation on the given parents.
        
        Args:
            parents: List of parent chromosomes (at least 2 required)
            
        Returns:
            offspring or list of offsprings: If number_offsprings=1, returns a single offspring.
                                            If number_offsprings>1, returns a list of offsprings.
        """
        if not parents or len(parents) < 2:
            raise ValueError("At least 2 parents are required")
        
        parent_num = len(parents)
        
        # Calculate crossover_num if not provided during initialization
        # We want at least parent_num crossover points (parent_num + 1 segments) for good mixing
        # But also respect the chromosome length constraints
        min_length = min(len(p) for p in parents)
        min_crossover = parent_num - 1  # Minimum to use each parent at least once
        max_crossover = min_length - 1  # Maximum based on shortest chromosome
        
        # Generate multiple offsprings
        offsprings = []
        for _ in range(self.number_offsprings):
            crossover_num = self.crossover_num if self.crossover_num is not None else random.randint(
                max(1, min_crossover), 
                max(1, max_crossover)
            )
            
            # Temporarily store crossover_num for _get_crossover_points
            original_crossover_num = self.crossover_num
            self.crossover_num = crossover_num
            
            # 1. Get crossover points for each parent (calculated independently for non-homologous lengths)
            all_points = [self._get_crossover_points(len(parent)) for parent in parents]
            
            # 2. Split each parent into segments
            all_segments = [self._split_chromosome(parent, points) 
                           for parent, points in zip(parents, all_points)]
            
            # 3. Rotate through segments from all parents to create offspring
            offspring = self._rotate_segments_multiparent(all_segments, parent_num)
            offsprings.append(offspring)
            
            # Restore original crossover_num
            self.crossover_num = original_crossover_num
        
        # Return single offspring if number_offsprings=1, otherwise return list
        return offsprings[0] if self.number_offsprings == 1 else offsprings


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: 3 Parents with 2 crossover points each")
    print("=" * 60)
    
    # Three parents of different lengths
    parent_1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
    parent_2 = ['a', 'b', 'c', 'd', 'e']
    parent_3 = ['X', 'Y', 'Z']
    
    # Setup crossover with 2 crossover points
    crossover = NCrossoverMultiparent(crossover_num=2)
    
    # Run crossover with 3 parents
    offspring = crossover.crossover([parent_1, parent_2, parent_3])
    
    print(f"Parent 1: {parent_1}")
    print(f"Parent 2: {parent_2}")
    print(f"Parent 3: {parent_3}")
    print("-" * 60)
    print(f"Offspring: {offspring}")
    print()
    
    # Explanation of the rotation:
    # Segment 0 (index 0 % 3 = 0) -> from Parent 1
    # Segment 1 (index 1 % 3 = 1) -> from Parent 2
    # Segment 2 (index 2 % 3 = 2) -> from Parent 3
    # Segment 3 (index 3 % 3 = 0) -> from Parent 1 (rotation back to first)
    # etc.
    
    print("=" * 60)
    print("Example 2: 4 Parents with segment_size=2")
    print("=" * 60)
    
    parent_1 = [1, 2, 3, 4, 5, 6, 7, 8]
    parent_2 = ['a', 'b', 'c', 'd', 'e', 'f']
    parent_3 = ['X', 'Y', 'Z', 'W']
    parent_4 = [10, 20, 30, 40, 50, 60, 70]
    
    # Using segment_size instead of crossover_num
    crossover = NCrossoverMultiparent(segment_size=2)
    
    offspring = crossover.crossover([parent_1, parent_2, parent_3, parent_4])
    
    print(f"Parent 1: {parent_1}")
    print(f"Parent 2: {parent_2}")
    print(f"Parent 3: {parent_3}")
    print(f"Parent 4: {parent_4}")
    print("-" * 60)
    print(f"Offspring: {offspring}")
    print()
    
    print("=" * 60)
    print("Example 3: 2 Parents (backward compatible)")
    print("=" * 60)
    
    parent_1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
    parent_2 = ['a', 'b', 'c', 'd', 'e']
    
    crossover = NCrossoverMultiparent(crossover_num=2)
    offspring = crossover.crossover([parent_1, parent_2])
    
    print(f"Parent 1: {parent_1}")
    print(f"Parent 2: {parent_2}")
    print("-" * 60)
    print(f"Offspring: {offspring}")
    print()
    
    print("=" * 60)
    print("Example 4: Generate 5 offsprings from 3 parents")
    print("=" * 60)
    
    parent_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    parent_2 = ['a', 'b', 'c', 'd', 'e', 'f']
    parent_3 = ['X', 'Y', 'Z', 'W']
    
    # Generate 5 offsprings
    crossover = NCrossoverMultiparent(crossover_num=3, number_offsprings=5)
    offsprings = crossover.crossover([parent_1, parent_2, parent_3])
    
    print(f"Parent 1: {parent_1}")
    print(f"Parent 2: {parent_2}")
    print(f"Parent 3: {parent_3}")
    print("-" * 60)
    for i, offspring in enumerate(offsprings, 1):
        print(f"Offspring {i}: {offspring}")