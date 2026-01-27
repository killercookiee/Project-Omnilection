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

import random

class NHomologousCrossover:
    def __init__(self, parent_1, parent_2, N_1_distribution=None, N_2_distribution=None):
        """
        Initialize N-Homologous Crossover
        
        Parameters:
        - parent_1: First parent chromosome (list of genes)
        - parent_2: Second parent chromosome (list of genes)
        - N_1_distribution: Function that returns number of crossover points for parent 1
        - N_2_distribution: Function that returns number of crossover points for parent 2
        """
        self.parent_1 = parent_1
        self.parent_2 = parent_2
        
        # Default uniform distribution: returns a random number between 1 and len(chromosome)-1
        if N_1_distribution is None:
            self.N_1_distribution = lambda: random.randint(1, max(1, len(parent_1) - 1))
        else:
            self.N_1_distribution = N_1_distribution
            
        if N_2_distribution is None:
            self.N_2_distribution = lambda: random.randint(1, max(1, len(parent_2) - 1))
        else:
            self.N_2_distribution = N_2_distribution
    
    def _get_crossover_points(self, chromosome_length, n_points):
        """
        Randomly select n_points crossover points for a chromosome
        
        Parameters:
        - chromosome_length: Length of the chromosome
        - n_points: Number of crossover points to select
        
        Returns:
        - Sorted list of crossover point indices
        """
        if chromosome_length <= 1:
            return []
        
        # Select random positions (can repeat)
        points = [random.randint(1, chromosome_length - 1) for _ in range(n_points)]
        return sorted(points)
    
    def _split_chromosome(self, chromosome, crossover_points):
        """
        Split chromosome into segments based on crossover points
        
        Parameters:
        - chromosome: The chromosome to split
        - crossover_points: Sorted list of crossover point indices
        
        Returns:
        - List of segments
        """
        if not crossover_points:
            return [chromosome]
        
        segments = []
        prev_point = 0
        
        for point in crossover_points:
            segments.append(chromosome[prev_point:point])
            prev_point = point
        
        # Add the last segment
        segments.append(chromosome[prev_point:])
        
        return segments
    
    def _merge_segments(self, segments_1, segments_2):
        """
        Merge corresponding segment pairs by randomly sampling genes
        
        Parameters:
        - segments_1: Segments from parent 1
        - segments_2: Segments from parent 2
        
        Returns:
        - Offspring chromosome
        """
        offspring = []
        
        # Process only up to min number of segments
        n_segments = min(len(segments_1), len(segments_2))
        
        for i in range(n_segments):
            seg_1 = segments_1[i]
            seg_2 = segments_2[i]
            
            # Determine the length to sample (up to the longer segment)
            max_len = max(len(seg_1), len(seg_2))
            
            for j in range(max_len):
                # Collect available genes at this position
                available_genes = []
                
                if j < len(seg_1):
                    available_genes.append(seg_1[j])
                if j < len(seg_2):
                    available_genes.append(seg_2[j])
                
                # Randomly sample from available genes
                if available_genes:
                    offspring.append(random.choice(available_genes))
        
        return offspring
    
    def crossover(self):
        """
        Perform N-homologous crossover
        
        Returns:
        - Two offspring chromosomes
        """
        # Get number of crossover points from distributions
        N_1 = self.N_1_distribution()
        N_2 = self.N_2_distribution()
        
        # Get crossover points for each parent
        crossover_points_1 = self._get_crossover_points(len(self.parent_1), N_1)
        crossover_points_2 = self._get_crossover_points(len(self.parent_2), N_2)
        
        # Split chromosomes into segments
        segments_1 = self._split_chromosome(self.parent_1, crossover_points_1)
        segments_2 = self._split_chromosome(self.parent_2, crossover_points_2)
        
        # Create two offspring by merging segments differently
        offspring_1 = self._merge_segments(segments_1, segments_2)
        offspring_2 = self._merge_segments(segments_2, segments_1)
        
        return offspring_1, offspring_2


# Example usage:
if __name__ == "__main__":
    # Example parents
    parent_1 = [1, 2, 3, 4, 5, 6, 7, 8]
    parent_2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    
    # Create crossover instance with default uniform distribution
    crossover = NHomologousCrossover(parent_1, parent_2)
    
    # Perform crossover
    offspring_1, offspring_2 = crossover.crossover()
    
    print("Parent 1:", parent_1)
    print("Parent 2:", parent_2)
    print("Offspring 1:", offspring_1)
    print("Offspring 2:", offspring_2)
    
    # Example with custom distribution
    def custom_distribution():
        return 3  # Always use 3 crossover points
    
    crossover_custom = NHomologousCrossover(
        parent_1, 
        parent_2, 
        N_1_distribution=custom_distribution,
        N_2_distribution=lambda: random.randint(2, 4)
    )
    
    offspring_1, offspring_2 = crossover_custom.crossover()
    print("\nWith custom distribution:")
    print("Offspring 1:", offspring_1)
    print("Offspring 2:", offspring_2)