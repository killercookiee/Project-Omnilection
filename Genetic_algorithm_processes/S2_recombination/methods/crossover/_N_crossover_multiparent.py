# Input N parent chromosomes (lists of genes)
# Pick N random crossover points on each parent chromosome to split the chromosomes into segments

# Offspring: Rotate through segments from all parents to create the offspring
# Order: P1_Seg1, P2_Seg2, P3_Seg3, ..., P1_Seg(N+1), P2_Seg(N+2), etc.

# Input parameters:
# - crossover_num: Number of crossover points per parent, determines size of segments | default is random
# - distribution: Function to return the crossover point depending on the length of the chromosome | default is uniform distribution
# - number_offsprings: Number of offsprings to produce (default is 1, but can be set to more for batch generation)

# Output:
# - offspring: One chromosome created by rotating segments from all parents

# Function input:
# - parents: List of parent chromosomes (each chromosome is a list of genes)



"""
Genetic_algorithm_processes/S2_recombination/methods/crossover/N_crossover_multiparent.py
"""

import random


class NCrossoverMultiparent: # ADVICE: take crossover_num as number of parents - 1
    def __init__(self,
        crossover_num: int = 1,
        distribution: callable = lambda length: random.randint(1, length),
        number_offsprings: int = 1,
        verbose: bool = False
    ):
        self.crossover_num = crossover_num
        self.number_offsprings = number_offsprings
        self.distribution = distribution
        self.verbose = verbose

        if self.verbose:
            print(f"[NCrossoverMultiparent] Initialized — crossover_num: {self.crossover_num} | offsprings: {self.number_offsprings}")

    def _get_crossover_points(self, chromosome_length: int) -> list[int]:
        n_points = self.crossover_num
        if n_points <= 0:
            return []

        points = set()
        attempts = 0
        max_attempts = n_points * 5

        while len(points) < n_points and attempts < max_attempts:
            p = self.distribution(chromosome_length)
            if 0 <= p <= chromosome_length:
                points.add(p)
            attempts += 1

        return sorted(list(points))

    def _split_chromosome(self, chromosome, crossover_points):
        if not crossover_points:
            return [chromosome]

        segments = []
        prev_point = 0
        for point in crossover_points:
            segments.append(chromosome[prev_point:point])
            prev_point = point
        segments.append(chromosome[prev_point:])
        
        return [seg for seg in segments if seg]

    def _rotate_segments_multiparent(self, all_segments, parent_num):
        offspring = []
        max_segments = max(len(segments) for segments in all_segments)

        for seg_idx in range(max_segments):
            parent_idx = seg_idx % parent_num
            if seg_idx < len(all_segments[parent_idx]):
                segment = all_segments[parent_idx][seg_idx]
                if segment:
                    offspring.extend(segment)

        return offspring

    def crossover(self, parents):
        if not parents or len(parents) < 2:
            raise ValueError("At least 2 parents are required")

        parent_num = len(parents)
        min_length = min(len(p) for p in parents)
        min_crossover = parent_num - 1
        max_crossover = min_length - 1

        if self.verbose:
            print(f"\n[NCrossoverMultiparent] Crossing {parent_num} parents → {self.number_offsprings} offspring")
            for i, p in enumerate(parents):
                print(f"  Parent [{i+1}] length: {len(p)}  preview: {p[:6]}{'...' if len(p) > 6 else ''}")

        offsprings = []
        for offspring_idx in range(self.number_offsprings):
            crossover_num = self.crossover_num if self.crossover_num is not None else random.randint(
                max(1, min_crossover),
                max(1, max_crossover)
            )

            original_crossover_num = self.crossover_num
            self.crossover_num = crossover_num

            all_points = [self._get_crossover_points(len(parent)) for parent in parents]
            all_segments = [self._split_chromosome(parent, points)
                           for parent, points in zip(parents, all_points)]
            offspring = self._rotate_segments_multiparent(all_segments, parent_num)
            offsprings.append(offspring)

            if self.verbose:
                SHADES = ['\033[47m', '\033[100m', '\033[107m', '\033[40m']  # one shade per parent
                RESET = '\033[0m'
                CELL = '  '  # two spaces per gene

                print(f"\n{'─'*40}")
                print(f"  Offspring [{offspring_idx+1}/{self.number_offsprings}]")

                # Each parent row: all segments share the same shade, gaps at cut points
                for i, (points, segments) in enumerate(zip(all_points, all_segments)):
                    shade = SHADES[i % len(SHADES)]
                    row = ""
                    for seg_idx, segment in enumerate(segments):
                        row += shade + CELL * len(segment) + RESET
                        if seg_idx < len(segments) - 1:
                            row += CELL  # gap at cut point
                    print(f"  P{i+1} │{row}│")

                print(f"  {'─'*36}")

                # Offspring row
                max_segs = max(len(s) for s in all_segments)
                offspring_row = ""
                non_empty_parts = []
                for seg_idx in range(max_segs):
                    src_parent = seg_idx % parent_num
                    if seg_idx < len(all_segments[src_parent]):
                        segment = all_segments[src_parent][seg_idx]
                        if segment:
                            shade = SHADES[src_parent % len(SHADES)]
                            non_empty_parts.append(shade + CELL * len(segment) + RESET)

                offspring_row = CELL.join(non_empty_parts)  # gap only BETWEEN segments, not after last
                print(f"  OS │{offspring_row}│")
                print(f"  Offspring genes: {offspring}")  # print actual genes
                print(f"{'─'*40}")
            
            self.crossover_num = original_crossover_num

        return offsprings[0] if self.number_offsprings == 1 else offsprings


if __name__ == "__main__":

    print("=" * 60)
    print("Example 1: 2 Parents, 1 crossover points")
    print("=" * 60)
    parent_1 = [1, 2, 3, 4, 5]
    parent_2 = ['a', 'b', 'c']
    crossover = NCrossoverMultiparent(crossover_num=1, number_offsprings=3, verbose=True)
    offspring = crossover.crossover([parent_1, parent_2])
    print(f"Final offspring: {offspring}\n")

    print("=" * 60)
    print("Example 2: 3 Parents, 3 crossover points")
    print("=" * 60)
    parent_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    parent_2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']
    parent_3 = ['X', 'Y', 'Z', 'W', 'V', 'U', 'T', 'S', 'R', 'Q']
    crossover = NCrossoverMultiparent(crossover_num=3, number_offsprings=3, verbose=True)
    offsprings = crossover.crossover([parent_1, parent_2, parent_3])
    print(f"Final — {len(offsprings)} offsprings generated")

    print("=" * 60)
    print("Example 3: Real scenario, 3 parents, 2 crossover points")
    print("=" * 60)
    selected_prompt_chains = [
        [("gemma3:270m", "Summarize the task and give a hint of the answer. ", "Make it concise and clear."),
         ("qwen2.5-coder:0.5b", "Provide 1 word answer only")],

        [("smollm:360m", "Summarize the task. ", "Give a hint of the completely wrong answer."),
         ("qwen2:0.5b", "Give a wrong answer. ", "Explain why the right answer is correct and the wrong answer is wrong.")],

        [("deepseek-coder:latest", "Try to give a wrong but sort of right answer. ")]
    ]
    crossover = NCrossoverMultiparent(crossover_num=2, number_offsprings=1, verbose=True)
    offspring = crossover.crossover(selected_prompt_chains)
    print(f"Final offspring: {offspring}")