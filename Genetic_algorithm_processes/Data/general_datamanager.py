"""
Genetic_algorithm_processes/Data/general_datamanager.py
"""

import json
import os
from datetime import datetime, timezone

from Genetic_algorithm_processes.Data.lineage_scoring import LineageScorer


# Example heritage data structure (in heritage_data.json):
# heritage_database = {
#     "generations": 5,
#     "version": "1.0",
#     "size": 2,
#     "prompt_chains": {
#         "prompt_chain_id_3": {
#             "parents": ["parent_prompt_chain_id_1", "parent_prompt_chain_id_2"],
#             "fitness": 0.8,
#             "generation": [1, 2, 3, 4, 5],
#             "metadata": {
#                 "creation_time": "2024-01-01T12:00:00Z",
#                 "modification_time": "2024-01-01T12:00:00Z",
#                 "other_info": "..."
#             }
#         },
#         "prompt_chain_id_4": {
#             "parents": ["parent_prompt_chain_id_3"],
#             "fitness": 0.6,
#             "generation": [4, 5],
#             "metadata": {
#                 "creation_time": "2024-01-02T12:00:00Z",
#                 "modification_time": "2024-01-02T12:00:00Z",
#                 "other_info": "..."
#             }
#         },
#         ...
#     }
# }


# Example population data structure (in population_data.json):
# population_data = {
#     "population": [
#         (prompt_chain_id_1, [("modelA", ["prompt_section_1", "prompt_section_2"]), ("modelB", ["prompt_section_3", "prompt_section_4"]), ...], fitness_score_1, metadata_1),
#         (prompt_chain_id_2, [("modelC", ["prompt_section_5", "prompt_section_6"]), ("modelD", ["prompt_section_7", "prompt_section_8"]), ...], fitness_score_2, metadata_2),
#         ...
#     ],
#     "metadata": {
#         "generation": 5,
#         "creation_time": "2024-01-01T12:00:00Z",
#         "modification_time": "2024-01-01T12:00:00Z",
#         "other_info": "..."
#     }
# }


# Example id_to_promptchain data structure:
# id, prompt_chain
# prompt_chain_id_1, [("modelA", ["prompt_section_1", "prompt_section_2"]), ("modelB", ["prompt_section_3", "prompt_section_4"]), ...]
# prompt_chain_id_2, ...


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class HeritageDataManager:
    def __init__(self, heritage_database_file: str):
        self.heritage_database_file = heritage_database_file
        self.local_heritage_database = self._load_heritage_database()

    def _load_heritage_database(self) -> dict:
        try:
            with open(self.heritage_database_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "generations": 0,
                "version": "1.0",
                "size": 0,
                "prompt_chains": {},
            }

    def _save_heritage_database(self) -> None:
        with open(self.heritage_database_file, "w") as f:
            json.dump(self.local_heritage_database, f, indent=4)

    def add_new(self, prompt_chain_id: str, parents: list[str], fitness: float | None, generation: list[int], metadata: dict = None) -> None:
        chains: dict = self.local_heritage_database.setdefault("prompt_chains", {})
        if prompt_chain_id in chains:
            return

        now = _now_iso()
        
        if metadata is None:
            metadata = {}
            
        metadata.setdefault("creation_time", now)
        metadata.setdefault("modification_time", now)

        chains[prompt_chain_id] = {
            "parents": parents,
            "fitness": fitness,
            "generation": generation,
            "metadata": metadata,
        }

        all_gens = [g for entry in chains.values() for g in entry.get("generation", [])]
        self.local_heritage_database["generations"] = max(all_gens) if all_gens else 0
        self.local_heritage_database["size"] = len(chains)
        self._save_heritage_database()

    def update_population(self, current_population: list[tuple]) -> None:
        chains: dict = self.local_heritage_database.setdefault("prompt_chains", {})
        now = _now_iso()

        for entry in current_population:
            prompt_chain_id, _prompt_chain, fitness, metadata = entry
            meta_generation = metadata.get("generation", []) if isinstance(metadata, dict) else []
            if isinstance(meta_generation, int):
                meta_generation = [meta_generation]

            if prompt_chain_id in chains:
                record = chains[prompt_chain_id]
                existing_gens: list = record.get("generation", [])
                for g in meta_generation:
                    if g not in existing_gens:
                        existing_gens.append(g)
                existing_gens.sort()
                record["fitness"] = fitness
                if isinstance(record.get("metadata"), dict):
                    record["metadata"]["modification_time"] = now
            else:
                chains[prompt_chain_id] = {
                    "parents": [],
                    "fitness": fitness,
                    "generation": meta_generation,
                    "metadata": metadata or {
                        "creation_time": now,
                        "modification_time": now,
                    },
                }

        all_gens = [g for rec in chains.values() for g in rec.get("generation", [])]
        self.local_heritage_database["generations"] = max(all_gens) if all_gens else 0
        self.local_heritage_database["size"] = len(chains)
        self._save_heritage_database()


class PopulationDataManager:
    def __init__(self, population_data_file: str):
        self.population_data_file = population_data_file
        self.local_population_data: dict | None = None
        self._load_population_data(population_data_file)

    def _load_population_data(self, file_path: str) -> None:
        try:
            with open(file_path, "r") as f:
                raw = json.load(f)
            population = [tuple(item) for item in raw.get("population", [])]
            self.local_population_data = {
                "population": population,
                "metadata": raw.get("metadata", {}),
            }
        except FileNotFoundError:
            self.local_population_data = {"population": [], "metadata": {}}

    def update_population_data(self, current_population: list) -> None:
        now = _now_iso()
        
        # 🚨 FIX: Sweep the entire population to find the true current generation
        max_generation = 0
        for entry in current_population:
            meta = entry[3]
            if isinstance(meta, dict):
                gen_val = meta.get("generation", 0)
                if isinstance(gen_val, list):
                    gen_val = max(gen_val) if gen_val else 0
                if gen_val > max_generation:
                    max_generation = gen_val

        self.local_population_data = {
            "population": [tuple(entry) for entry in current_population],
            "metadata": {
                "generation": max_generation,
                "creation_time": now,
                "modification_time": now,
            },
        }
        with open(self.population_data_file, "w") as f:
            serialisable_population = [list(entry) for entry in self.local_population_data["population"]]
            json.dump({"population": serialisable_population, "metadata": self.local_population_data["metadata"]}, f, indent=4)

class IDToPromptChainManager:
    def __init__(self, id_to_promptchain_file: str):
        self.id_to_promptchain_file = id_to_promptchain_file
        self._mapping: dict = self._load()

    def _load(self) -> dict:
        try:
            with open(self.id_to_promptchain_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def get_promptchain_from_id(self, prompt_chain_id: str):
        return self._mapping.get(prompt_chain_id)

    def add_or_update(self, prompt_chain_id: str, prompt_chain: list) -> None:
        self._mapping[prompt_chain_id] = prompt_chain
        with open(self.id_to_promptchain_file, "w") as f:
            json.dump(self._mapping, f, indent=4)


class EmbeddingCacheManager:
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self._cache: dict = self._load()

    def _load(self) -> dict:
        try:
            with open(self.cache_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save(self) -> None:
        with open(self.cache_file, "w") as f:
            json.dump(self._cache, f)


class GeneralDataManager:
    def __init__(self, run_dir: str = None, resume_from_save: bool = False):
        """
        Coordinates individual data-managers under a unified facade.
        Dynamically handles folder isolation and serialization overhead.
        """
        # ── Automated Directory Routing ──
        if resume_from_save and run_dir:
            self.run_dir = run_dir
            print(f"[GeneralDataManager] 💾 Resuming state from target directory: {self.run_dir}")
        else:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
            self.run_dir = f"test_run_{timestamp}"
            print(f"[GeneralDataManager] 📁 Initializing pristine run directory: {self.run_dir}")

        os.makedirs(self.run_dir, exist_ok=True)

        # Map filesystem bounds inside the designated run folder
        heritage_path = os.path.join(self.run_dir, "heritage_data.json")
        population_path = os.path.join(self.run_dir, "population_data.json")
        id_path = os.path.join(self.run_dir, "id_to_promptchain.json")
        emb_path = os.path.join(self.run_dir, "embedding_cache.json")

        self.heritage_data_manager = HeritageDataManager(heritage_path)
        self.population_data_manager = PopulationDataManager(population_path)
        self.id_to_promptchain_manager = IDToPromptChainManager(id_path)
        self.embedding_cache_manager = EmbeddingCacheManager(emb_path)
        
        self._lineage_cache: dict[str, float] = {}

    def register_new_chain(self, prompt_chain_id: str, prompt_chain: list, parents: list[str], fitness: float | None, generation: list[int], metadata: dict = None) -> None:
        self.id_to_promptchain_manager.add_or_update(prompt_chain_id, prompt_chain)
        self.heritage_data_manager.add_new(prompt_chain_id, parents, fitness, generation, metadata)

    def register_intermediary_chain(self, prompt_chain_id: str, prompt_chain: list, parents: list[str], metadata: dict = None) -> None:
        """Logs genetic material to the heritage tracker WITHOUT adding it to the active generation."""
        self.id_to_promptchain_manager.add_or_update(prompt_chain_id, prompt_chain)
        self.heritage_data_manager.add_new(
            prompt_chain_id=prompt_chain_id,
            parents=parents,
            fitness=None,  # Null fitness indicates it has not been formally evaluated
            generation=[], # Empty list indicates it hasn't survived into an active generation yet
            metadata=metadata
        )

    def sync_population(self, current_population: list) -> None:
        for entry in current_population:
            self.id_to_promptchain_manager.add_or_update(entry[0], entry[1])
        self.population_data_manager.update_population_data(current_population)
        self.heritage_data_manager.update_population(current_population)

    def get_chain(self, prompt_chain_id: str):
        return self.id_to_promptchain_manager.get_promptchain_from_id(prompt_chain_id)

    def refresh_lineage_scores(self, use_embeddings: bool = True) -> None:
        """Triggers the Prefix Tree Lineage Scorer against the database history."""
        scorer = LineageScorer(
            heritage_data=self.heritage_data_manager.local_heritage_database,
            id_to_chain=self.id_to_promptchain_manager._mapping,
            use_embeddings=use_embeddings,
            emb_cache=self.embedding_cache_manager._cache,
            verbose=False
        )
        self._lineage_cache = scorer.compute_all_lineage_scores()
        self.embedding_cache_manager.save()

    def lineage_score(self, prompt_chain_id: str) -> float:
        """Returns the pre-calculated lineage score mapped via Prefix Tree math."""
        return self._lineage_cache.get(prompt_chain_id, 0.0)

    def get_population_with_lineage(self, selected_chains: list[list[tuple]]) -> list[tuple]:
        """
        Formats raw population vectors into the contextual tuple pool required by 
        LineagePairing. Matches parent structures purely using structural prefixes.
        
        Returns: list of (chain_tuple, fitness, lineage_score, family_id)
        """
        chain_str_to_id = {str(chain): cid for cid, chain in self.id_to_promptchain_manager._mapping.items()}
        heritage_chains = self.heritage_data_manager.local_heritage_database.get("prompt_chains", {})
        
        pop_with_lineage = []
        for chain in selected_chains:
            cid = chain_str_to_id.get(str(chain))
            if cid and cid in heritage_chains:
                record = heritage_chains[cid]
                fitness = record.get("fitness", 0.0)
                # Ensure fitness is not None before passing to pairer
                if fitness is None:
                    continue
                lf_score = self.lineage_score(cid)
                family_id = record.get("metadata", {}).get("family_id", cid)
                pop_with_lineage.append((chain, float(fitness), lf_score, family_id))
        return pop_with_lineage