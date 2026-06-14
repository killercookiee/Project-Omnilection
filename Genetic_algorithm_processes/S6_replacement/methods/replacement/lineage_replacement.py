"""
Genetic_algorithm_processes/S6_replacement/methods/replacement/lineage_replacement.py
"""

class LineageReplacement:
    def __init__(self,
        population_cap: int = 10,
        exploration_ratio: float = 0.3,
        maturity_threshold: int = 3,
        verbose: bool = False
    ):
        self.population_cap = population_cap
        self.exploration_ratio = exploration_ratio
        self.maturity_threshold = maturity_threshold
        self.verbose = verbose

        if self.verbose:
            print(f"[LineageReplacement] Initialized — Cap: {self.population_cap} | Exploration Ratio: {self.exploration_ratio:.0%} | Maturity: {self.maturity_threshold} runs")

    def replace(self, combined_population_with_meta: list[tuple]) -> list[tuple]:
        """
        combined_population_with_meta: list of tuples
        (chain, fitness, lineage_score, family_id, family_size)
        """
        if not combined_population_with_meta:
            return []

        # 1. Remove exact duplicates (same exact chain) to maintain strict genetic diversity
        unique_population = []
        seen_chains = set()
        for ind in combined_population_with_meta:
            chain_str = str(ind[0])
            if chain_str not in seen_chains:
                unique_population.append(ind)
                seen_chains.add(chain_str)

        # 2. Categorize into Young (Explorers) and Mature (Elites)
        young_pool = []
        mature_pool = []
        
        for ind in unique_population:
            family_size = ind[4]
            if family_size < self.maturity_threshold:
                young_pool.append(ind)
            else:
                mature_pool.append(ind)

        # 3. Sort both pools by raw fitness (Highest first)
        young_pool.sort(key=lambda x: x[1], reverse=True)
        mature_pool.sort(key=lambda x: x[1], reverse=True)

        # 4. Determine Quotas
        explorer_quota = int(self.population_cap * self.exploration_ratio)
        elite_quota = self.population_cap - explorer_quota

        new_population = []

        # 5. Fill Explorer Quota (Keep best young chains)
        explorers_kept = young_pool[:explorer_quota]
        new_population.extend(explorers_kept)
        
        # Any unused explorer slots spill over to the elite quota
        spillover = explorer_quota - len(explorers_kept)
        elite_quota += spillover

        # 6. Fill Elite Quota (Keep best mature chains)
        # This naturally kills High-Lineage/Low-Fitness chains because they are sorted out
        elites_kept = mature_pool[:elite_quota]
        new_population.extend(elites_kept)

        # 7. Final fallback: If we still haven't hit the cap (rare), grab the best remaining young chains
        if len(new_population) < self.population_cap:
            remaining_young = young_pool[len(explorers_kept):]
            new_population.extend(remaining_young[:self.population_cap - len(new_population)])

        # Final sort by fitness for clean output
        new_population.sort(key=lambda x: x[1], reverse=True)

        if self.verbose:
            scores = [ind[1] for ind in new_population]
            print(f"\n{'─'*40}")
            print(f"  ✅  Lineage Replacement complete")
            print(f"      Kept       : {len(new_population)} / {len(combined_population_with_meta)} distinct chains")
            print(f"      Elites     : {len(elites_kept)}")
            print(f"      Explorers  : {len(explorers_kept)}")
            if scores:
                print(f"      Best       : {scores[0]:.3f}  |  Worst kept: {scores[-1]:.3f}  |  Avg: {sum(scores)/len(scores):.3f}")
            print(f"{'─'*40}\n")

        # Strip the extra metadata (lineage, family_id, size) before returning to the main GA pipeline
        # The main pipeline expects: (chain_id, chain, fitness, metadata)
        # We will handle this packaging in the wrapper class.
        return new_population