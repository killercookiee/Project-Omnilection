"""
Dataset_Prompts/initial_population_generator.py

Builds a semantically diverse initial population of prompt chains for the
Genetic Algorithm using sentence embeddings + K-Means clustering.

Diversity is enforced at three levels:
  1. Intra-step      – segments within one step are drawn from the same cluster
                       (semantic coherence per model call).
  2. Intra-chain     – every step in a single chain is drawn from a *different*
                       cluster (no two steps are semantically similar).
  3. Inter-individual – a candidate chain is accepted only when its mean
                        embedding is below a cosine-similarity ceiling versus
                        every already-accepted individual.

Output format (matches the rest of the GA pipeline):
    prompt_chain = [
        (model_name, [segment_1, segment_2, …]),   # step 1
        (model_name, [segment_3, …]),               # step 2
        …
    ]
"""

from __future__ import annotations

import os
import json
import hashlib
import math
import random
import warnings
from typing import Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is required. "
        "Install with: pip install sentence-transformers"
    )

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
except ImportError:
    raise ImportError(
        "scikit-learn is required. "
        "Install with: pip install scikit-learn"
    )


class InitialPopulationGenerator:
    """
    Generates a semantically diverse initial population of prompt chains.

    Parameters
    ----------
    segments : list[str]
        All cleaned prompt segments from ``GenePoolManager.load_prompt_segments()``.
    available_models : list[str]
        Model names assignable to each chain step (e.g. ``['gpt-4', 'gpt-3.5-turbo']``).
    population_cap : int
        Full GA population capacity.  The initial population will contain
        ``ceil(initial_pop_ratio * population_cap)`` individuals.
    n_clusters : int, optional
        Number of K-Means clusters.
        Defaults to ``min(80, max(10, floor(√N)))`` where N = len(segments).
    initial_pop_ratio : float
        Fraction of ``population_cap`` to seed.  Default: 0.2 (20 %).
    min_chain_length : int
        Minimum number of (model + prompts) steps per chain.  Default: 2.
    max_chain_length : int
        Maximum number of steps per chain.  Default: 4.
    min_segments_per_step : int
        Minimum prompt segments inside a single step.  Default: 1.
    max_segments_per_step : int
        Maximum prompt segments inside a single step.  Default: 3.
    diversity_threshold : float
        Cosine-similarity ceiling between any two accepted individuals' mean
        embeddings.  Lower → stricter diversity.  Range [0, 1].  Default: 0.85.
    embedding_model_name : str
        ``sentence-transformers`` model for encoding.
        Default: ``'all-MiniLM-L6-v2'`` (fast, 384-d, good quality).
    max_retries : int
        Per-individual retry budget before force-accepting.  Default: 30.
    random_seed : int, optional
        Seed for reproducible results.
    """

    def __init__(
        self,
        segments: list[str],
        available_models: list[str],
        population_cap: int,
        n_clusters: Optional[int] = None,
        initial_pop_ratio: float = 0.2,
        min_chain_length: int = 1,
        max_chain_length: int = 1,
        min_segments_per_step: int = 1,
        max_segments_per_step: int = 3,
        diversity_threshold: float = 0.85,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        max_retries: int = 30,
        random_seed: Optional[int] = None,
    ) -> None:
        if not segments:
            raise ValueError("Segment list must not be empty.")
        if not available_models:
            raise ValueError("At least one model name must be provided.")

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.segments = segments
        self.available_models = available_models
        self.population_cap = population_cap
        self.initial_pop_ratio = initial_pop_ratio
        self.target_size = max(1, math.ceil(population_cap * initial_pop_ratio))
        self.min_chain_length = min_chain_length
        self.max_chain_length = max_chain_length
        self.min_segments_per_step = min_segments_per_step
        self.max_segments_per_step = max_segments_per_step
        self.diversity_threshold = diversity_threshold
        self.max_retries = max_retries

        # Default cluster count: √N clamped to [10, 80]
        n = len(segments)
        default_k = min(80, max(10, int(math.sqrt(n))))
        self.n_clusters = min(
            n_clusters if n_clusters is not None else default_k,
            n,  # can never exceed the number of segments
        )

        print(
            f"[InitialPopulationGenerator] {n:,} segments | "
            f"{self.n_clusters} clusters | "
            f"target {self.target_size} individuals "
            f"({initial_pop_ratio:.0%} of cap={population_cap})"
        )

        self._stmodel = SentenceTransformer(embedding_model_name)

        # Populated lazily by _embed_and_cluster()
        self.embeddings: Optional[np.ndarray] = None        # (N, D) L2-normalised
        self.cluster_labels: Optional[np.ndarray] = None    # (N,)  int
        self.cluster_to_indices: dict[int, list[int]] = {}  # cluster_id → [seg_idx, …]

    # ------------------------------------------------------------------
    # Embedding & clustering
    # ------------------------------------------------------------------

    def _embed_and_cluster(self) -> None:
        """
        Encode every segment with the sentence-transformer model and assign
        each to a K-Means cluster. Uses local disk caching to prevent
        re-embedding a massive static gene pool on new runs.
        """
        # Create a deterministic hash of the entire segment pool
        pool_str = json.dumps(self.segments)
        pool_hash = hashlib.md5(pool_str.encode('utf-8')).hexdigest()[:12]
        
        # Store these globally alongside the dataset, NOT in the specific run folder
        cache_dir = "Dataset_Prompts/.cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Include n_clusters in the cluster filename so changing K regenerates the clusters
        emb_file = os.path.join(cache_dir, f"embeddings_{pool_hash}.npy")
        clus_file = os.path.join(cache_dir, f"clusters_{pool_hash}_k{self.n_clusters}.npy")

        if os.path.exists(emb_file) and os.path.exists(clus_file):
            print(f"[InitialPopulationGenerator] 💾 Loading cached embeddings and clusters from {cache_dir}...")
            self.embeddings = np.load(emb_file)
            self.cluster_labels = np.load(clus_file)
        else:
            print("[InitialPopulationGenerator] Encoding segments (This will only happen ONCE) …")
            raw_embeddings = self._stmodel.encode(
                self.segments,
                show_progress_bar=True,
                batch_size=256,
                convert_to_numpy=True,
            )
            self.embeddings = normalize(raw_embeddings, norm="l2")

            print("[InitialPopulationGenerator] K-Means clustering …")
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
            self.cluster_labels = kmeans.fit_predict(self.embeddings)

            # Save to cache as fast binary numpy files
            np.save(emb_file, self.embeddings)
            np.save(clus_file, self.cluster_labels)
            print(f"[InitialPopulationGenerator] 💾 Saved embeddings and clusters to {cache_dir}.")

        self.cluster_to_indices = {}
        for seg_idx, cluster_id in enumerate(self.cluster_labels):
            self.cluster_to_indices.setdefault(int(cluster_id), []).append(seg_idx)

        sizes = [len(v) for v in self.cluster_to_indices.values()]
        print(
            f"[InitialPopulationGenerator] Cluster stats — "
            f"min={min(sizes)}  max={max(sizes)}  "
            f"mean={sum(sizes) / len(sizes):.1f}  total_clusters={len(sizes)}"
        )

    # ------------------------------------------------------------------
    # Diversity helpers
    # ------------------------------------------------------------------

    def _mean_embedding(self, seg_indices: list[int]) -> np.ndarray:
        """
        Compute the mean of the L2-normalised embeddings for *seg_indices*,
        then renormalise.  Used as a compact fingerprint for diversity checks.
        """
        vecs = self.embeddings[seg_indices]   # (K, D)
        mean = vecs.mean(axis=0)              # (D,)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 1e-9 else mean

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalised vectors."""
        return float(np.dot(a, b))

    def _is_diverse_enough(
        self,
        candidate_emb: np.ndarray,
        accepted_embs: list[np.ndarray],
    ) -> bool:
        """
        Return True iff *candidate_emb* is below the similarity ceiling
        against every already-accepted individual.
        """
        return all(
            self._cosine(candidate_emb, acc_emb) <= self.diversity_threshold
            for acc_emb in accepted_embs
        )

    # ------------------------------------------------------------------
    # Chain construction
    # ------------------------------------------------------------------

    def _sample_chain(self) -> tuple[list[tuple], list[int]]:
        """
        Build one candidate prompt chain.

        Intra-chain diversity is enforced by assigning each step to a
        *different* K-Means cluster (semantically distinct prompt groups).

        Returns
        -------
        chain : list[tuple]
            ``[(model, [seg_1, seg_2, …]), …]``
        all_seg_indices : list[int]
            Every segment index used across all steps (for embedding lookup).
        """
        chain_length = random.randint(self.min_chain_length, self.max_chain_length)

        # Shuffle and pick `chain_length` distinct cluster IDs.
        # If there are fewer clusters than desired steps, wrap around (still
        # maximally diverse within the available cluster budget).
        all_cluster_ids = list(self.cluster_to_indices.keys())
        random.shuffle(all_cluster_ids)

        if len(all_cluster_ids) >= chain_length:
            chosen_clusters = all_cluster_ids[:chain_length]
        else:
            repeats = math.ceil(chain_length / len(all_cluster_ids))
            chosen_clusters = (all_cluster_ids * repeats)[:chain_length]

        chain: list[tuple] = []
        all_seg_indices: list[int] = []

        for cluster_id in chosen_clusters:
            model = random.choice(self.available_models)

            # Sample 1–max segments from this cluster
            pool = self.cluster_to_indices[cluster_id]
            n_segs = min(
                random.randint(self.min_segments_per_step, self.max_segments_per_step),
                len(pool),
            )
            chosen_seg_indices = random.sample(pool, n_segs)
            all_seg_indices.extend(chosen_seg_indices)

            step_segments = [self.segments[i] for i in chosen_seg_indices]
            # Step format: (model_name, [segment_1, segment_2, …])
            chain.append((model, step_segments))

        return chain, all_seg_indices

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> list[list[tuple]]:
        """
        Generate the semantically diverse initial population.

        Runs embedding + clustering on first call, then iteratively builds
        candidate chains, rejecting any that are too similar to already-accepted
        individuals.  If ``max_retries`` is exhausted for a single slot, the
        current best candidate is force-accepted with a warning (and the
        diversity_threshold can be tuned to avoid this).

        Returns
        -------
        list[list[tuple]]
            A list of ``target_size`` prompt chains, each a list of
            ``(model_name, [segment_1, segment_2, …])`` step-tuples.
        """
        self._embed_and_cluster()

        population: list[list[tuple]] = []
        accepted_embs: list[np.ndarray] = []
        retries_for_current_slot = 0
        total_attempts = 0

        while len(population) < self.target_size:
            total_attempts += 1
            chain, seg_indices = self._sample_chain()
            candidate_emb = self._mean_embedding(seg_indices)

            if self._is_diverse_enough(candidate_emb, accepted_embs):
                population.append(chain)
                accepted_embs.append(candidate_emb)
                retries_for_current_slot = 0
                print(
                    f"[InitialPopulationGenerator] "
                    f"  ✓ individual {len(population)}/{self.target_size} "
                    f"(attempt #{total_attempts})"
                )
            else:
                retries_for_current_slot += 1
                if retries_for_current_slot >= self.max_retries:
                    warnings.warn(
                        f"Diversity threshold ({self.diversity_threshold}) could not be "
                        f"satisfied after {self.max_retries} retries — force-accepting "
                        f"current candidate.  Consider lowering `diversity_threshold` "
                        f"or increasing `n_clusters`.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    population.append(chain)
                    accepted_embs.append(candidate_emb)
                    retries_for_current_slot = 0

        print(
            f"[InitialPopulationGenerator] Generation complete — "
            f"{len(population)} individuals accepted in {total_attempts} total attempts."
        )
        return population