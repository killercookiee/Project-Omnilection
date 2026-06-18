"""
Genetic_algorithm_processes/S3_mutation/methods/gene_pool_search_mutation.py

GenePoolSearchMutation
-----------------------
A mutation operator that swaps a segment in a prompt chain for a *semantically
related* segment pulled from the (huge) gene pool, instead of asking an LLM to
rewrite text from scratch (that's what SemanticLLMMutation already does).

Why this exists
~~~~~~~~~~~~~~~
The gene pool (`prompt_segments.yaml`) can contain tens of thousands of
segments. Naively scanning/embedding all of them on every mutation call would
be far too slow to run every generation. Instead, this operator:

  1. Reuses the K-Means clustering already computed (and cached to disk) by
     `InitialPopulationGenerator` — no re-embedding of the whole gene pool,
     no new clustering pass. We just load the cached `.npy` artifacts.
  2. Embeds *only* the single target segment being mutated, finds its
     nearest cluster centroid, and samples a replacement candidate from
     that cluster only. This keeps the search fast (one embedding call)
     and topically scoped (same cluster = semantically related).
  3. Hard-blocks the swap with regex-based structural checks so a "lucky"
     semantic match can't silently strip out role/format scaffolding and
     break the chain's I/O contract. If the original segment carries
     role/format instructions, only candidates that preserve an equivalent
     structural signature are eligible; otherwise the swap is skipped.

This keeps the search "broad enough to matter" (full gene pool is in scope
via its cluster) while staying "narrow enough to be fast and safe" (one
cluster, regex-gated).
"""

from __future__ import annotations

import os
import re
import json
import hashlib
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


# ──────────────────────────────────────────────────────────────────────────
# Structural / format guard
# ──────────────────────────────────────────────────────────────────────────
#
# These patterns flag segments that carry "chain logic" — i.e. text that
# defines a role, or specifies the input/output contract for a step. If the
# *original* segment trips one of these categories, a replacement candidate
# must trip the SAME category (or be otherwise structurally compatible) or
# the swap is rejected outright.

_ROLE_PATTERNS = [
    re.compile(r"\byou are\b", re.IGNORECASE),
    re.compile(r"\bact as\b", re.IGNORECASE),
    re.compile(r"\byour role\b", re.IGNORECASE),
    re.compile(r"\bas an? (ai|assistant|expert|agent)\b", re.IGNORECASE),
]

_OUTPUT_FORMAT_PATTERNS = [
    re.compile(r"\boutput\s*(format|:)\b", re.IGNORECASE),
    re.compile(r"\brespond\s+(only\s+)?(with|in)\b", re.IGNORECASE),
    re.compile(r"\breturn\s+(only\s+)?(a|the|valid)?\s*(json|xml|yaml|csv)\b", re.IGNORECASE),
    re.compile(r"\bprovide only\b", re.IGNORECASE),
    re.compile(r"\bdo not (include|add|explain)\b", re.IGNORECASE),
    re.compile(r"```"),                       # fenced code / schema blocks
    re.compile(r"\{[^{}]{0,80}\}"),            # {placeholder} / json-ish braces
    re.compile(r"^\s*(input|output)\s*:", re.IGNORECASE | re.MULTILINE),
]

_CONSTRAINT_PATTERNS = [
    re.compile(r"\bmust\b", re.IGNORECASE),
    re.compile(r"\bonly\b", re.IGNORECASE),
    re.compile(r"\bnever\b", re.IGNORECASE),
    re.compile(r"\balways\b", re.IGNORECASE),
]


def _signature(text: str) -> tuple[bool, bool, bool]:
    """
    Boolean structural fingerprint of a segment: (has_role, has_output_format,
    has_hard_constraint). Used to decide whether a candidate is allowed to
    replace a given original segment.
    """
    has_role = any(p.search(text) for p in _ROLE_PATTERNS)
    has_format = any(p.search(text) for p in _OUTPUT_FORMAT_PATTERNS)
    has_constraint = any(p.search(text) for p in _CONSTRAINT_PATTERNS)
    return has_role, has_format, has_constraint


def _is_structurally_safe_swap(original: str, candidate: str) -> bool:
    """
    Returns True iff replacing `original` with `candidate` will not strip out
    role / output-format / hard-constraint scaffolding that the chain logic
    depends on.

    Rule: for each structural category the ORIGINAL segment trips, the
    CANDIDATE must trip it too. Candidates are free to *add* structure the
    original didn't have (that's just more explicit, not chain-breaking),
    but they can never silently *remove* role/format/constraint logic that
    was there.
    """
    orig_role, orig_format, orig_constraint = _signature(original)
    cand_role, cand_format, cand_constraint = _signature(candidate)

    if orig_role and not cand_role:
        return False
    if orig_format and not cand_format:
        return False
    if orig_constraint and not cand_constraint:
        return False
    return True


class GenePoolSearchMutation:
    """
    Mutation operator: replace one segment of one step with a semantically
    related segment pulled from the gene pool's cached K-Means clusters.

    Parameters
    ----------
    segments : list[str]
        The full, cleaned gene pool (same list passed into
        `InitialPopulationGenerator`). Needed so we can map cluster indices
        back to actual segment text and so the cache lookup hash matches.
    embedding_model_name : str
        Must match the model used by `InitialPopulationGenerator` so the
        cached embeddings/clusters are compatible. Default: 'all-MiniLM-L6-v2'.
    cache_dir : str
        Directory where `InitialPopulationGenerator` wrote its cached
        embeddings/clusters. Default: 'Dataset_Prompts/.cache'.
    n_clusters : int, optional
        The K used when the cache was built. If you don't know it, leave
        None and the operator will auto-discover the cluster file that
        matches the segment-pool hash (works as long as exactly one K was
        cached for this pool; pass it explicitly if you cached multiple Ks).
    max_candidate_attempts : int
        How many same-cluster candidates to try before giving up on this
        mutation call. Default: 10.
    min_segment_length : int
        Segments shorter than this are not worth mutating. Default: 5.
    verbose : bool
    """

    def __init__(
        self,
        segments: list[str],
        embedding_model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = "Dataset_Prompts/.cache",
        n_clusters: Optional[int] = None,
        max_candidate_attempts: int = 10,
        min_segment_length: int = 5,
        verbose: bool = False,
    ) -> None:
        if not segments:
            raise ValueError("Segment list must not be empty.")

        self.segments = segments
        self.cache_dir = cache_dir
        self.max_candidate_attempts = max_candidate_attempts
        self.min_segment_length = min_segment_length
        self.verbose = verbose

        self._stmodel = SentenceTransformer(embedding_model_name)

        self.embeddings: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_to_indices: dict[int, list[int]] = {}

        self._load_cached_clusters(n_clusters)

    # ------------------------------------------------------------------
    # Cache loading (no re-embedding, no re-clustering — reuse only)
    # ------------------------------------------------------------------

    def _load_cached_clusters(self, n_clusters: Optional[int]) -> None:
        pool_str = json.dumps(self.segments)
        pool_hash = hashlib.md5(pool_str.encode("utf-8")).hexdigest()[:12]

        emb_file = os.path.join(self.cache_dir, f"embeddings_{pool_hash}.npy")

        if n_clusters is not None:
            clus_file = os.path.join(self.cache_dir, f"clusters_{pool_hash}_k{n_clusters}.npy")
        else:
            # Auto-discover any cached cluster file for this exact pool hash.
            clus_file = None
            if os.path.isdir(self.cache_dir):
                pattern = re.compile(rf"^clusters_{pool_hash}_k(\d+)\.npy$")
                candidates = []
                for fname in os.listdir(self.cache_dir):
                    m = pattern.match(fname)
                    if m:
                        candidates.append((int(m.group(1)), fname))
                if candidates:
                    # Prefer the largest K available (finer-grained clusters).
                    candidates.sort(reverse=True)
                    clus_file = os.path.join(self.cache_dir, candidates[0][1])

        if not clus_file or not os.path.exists(emb_file) or not os.path.exists(clus_file):
            raise FileNotFoundError(
                "GenePoolSearchMutation requires the embeddings/clusters cache "
                "already built by InitialPopulationGenerator. Run "
                "InitialPopulationGenerator.generate() at least once (with the "
                "SAME segment list and embedding model) before constructing "
                f"this mutation operator. Looked for cache in '{self.cache_dir}' "
                f"matching pool hash '{pool_hash}'."
            )

        if self.verbose:
            print(f"[GenePoolSearchMutation] 💾 Reusing cached clusters from {clus_file}")

        self.embeddings = np.load(emb_file)
        self.cluster_labels = np.load(clus_file)

        self.cluster_to_indices = {}
        for seg_idx, cluster_id in enumerate(self.cluster_labels):
            self.cluster_to_indices.setdefault(int(cluster_id), []).append(seg_idx)

        # Precompute cluster centroids (mean of L2-normalised member embeddings,
        # renormalised) purely so we can snap a freshly-embedded query segment
        # to its nearest existing cluster without re-running K-Means.
        self._centroids: dict[int, np.ndarray] = {}
        for cluster_id, idxs in self.cluster_to_indices.items():
            vecs = self.embeddings[idxs]
            mean = vecs.mean(axis=0)
            norm = np.linalg.norm(mean)
            self._centroids[cluster_id] = mean / norm if norm > 1e-9 else mean

    # ------------------------------------------------------------------
    # Cluster lookup for an arbitrary (possibly mutated) query segment
    # ------------------------------------------------------------------

    def _nearest_cluster(self, text: str) -> int:
        query_emb = self._stmodel.encode([text], convert_to_numpy=True)[0]
        norm = np.linalg.norm(query_emb)
        if norm > 1e-9:
            query_emb = query_emb / norm

        best_cluster, best_sim = None, -2.0
        for cluster_id, centroid in self._centroids.items():
            sim = float(np.dot(query_emb, centroid))
            if sim > best_sim:
                best_sim, best_cluster = sim, cluster_id
        return best_cluster

    # ------------------------------------------------------------------
    # Public mutation API — matches the (chain) -> chain contract used by
    # SemanticLLMMutation / DeleteMutation so it can be dropped straight
    # into PromptChainMutation.mutation_methods.
    # ------------------------------------------------------------------

    def mutate(self, chain: list) -> list:
        # ── Genetic Armor: Ensure chain is actually a list ──
        if not chain or not isinstance(chain, list):
            return chain

        step_idx = random.randint(0, len(chain) - 1)
        step = chain[step_idx]

        # ── Genetic Armor: Protect against malformed tuples ──
        if not isinstance(step, (list, tuple)) or len(step) < 2:
            return chain

        model_name = step[0]
        segments = step[1]

        # ── Genetic Armor: Ensure segments is a list of strings ──
        if isinstance(segments, str):
            segments = [segments]
        elif not isinstance(segments, list) or len(segments) == 0:
            return chain

        seg_idx = random.randint(0, len(segments) - 1)
        target_segment = str(segments[seg_idx])

        if len(target_segment.strip()) < self.min_segment_length:
            return chain

        # 1. Find the cluster this segment currently lives closest to.
        cluster_id = self._nearest_cluster(target_segment)
        pool_idxs = self.cluster_to_indices.get(cluster_id, [])
        if not pool_idxs:
            return chain

        if self.verbose:
            print(
                f"  [GenePoolSearch] target segment -> cluster {cluster_id} "
                f"({len(pool_idxs)} candidates available)"
            )

        # 2. Try same-cluster candidates until one passes the structural guard.
        tried: set[int] = set()
        attempts = min(self.max_candidate_attempts, len(pool_idxs))

        for _ in range(attempts):
            remaining = [i for i in pool_idxs if i not in tried]
            if not remaining:
                break

            cand_idx = random.choice(remaining)
            tried.add(cand_idx)
            candidate_text = self.segments[cand_idx]

            if not candidate_text or candidate_text.strip() == target_segment.strip():
                continue

            if not _is_structurally_safe_swap(target_segment, candidate_text):
                if self.verbose:
                    print("    [GenePoolSearch] rejected candidate — would strip role/format/constraint")
                continue

            # 3. Accepted — rebuild the chain with the new genetic material.
            new_segments = list(segments)
            new_segments[seg_idx] = candidate_text

            new_chain = list(chain)
            new_chain[step_idx] = (model_name, new_segments)

            if self.verbose:
                print(f"    [GenePoolSearch] ✓ swapped segment (cluster {cluster_id})")

            return new_chain

        # No structurally-safe candidate found in this cluster within budget.
        if self.verbose:
            print("    [GenePoolSearch] no safe candidate found — chain unchanged")
        return chain