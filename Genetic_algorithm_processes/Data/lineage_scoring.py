"""
Genetic_algorithm_processes/Data/lineage_scoring.py

Computes normalised lineage scores [0, 1] for every chain in the population
using a dynamic Prefix Tree (Trie) architecture.

Mathematical framework
──────────────────────
Lineage_raw = (μ − α·SE) + β·D·W

  μ    weighted mean fitness of all chains starting with this prefix
  SE   σ / √n_eff_clamped — uncertainty / approximation error 
  D    destiny score ∈ [0, 1] (Information Gain from prefix)
  W    Shapiro–Wilk statistic ∈ [0, 1] — normality proxy
  α, β dynamic hyperparameters auto-adjusted from population state

Prefix Tree Architecture
────────────────────────
A 'family' is dynamically defined by the exact sequence of prompts.
If base_chain = [A, B], its members are every chain in history that starts
with [A, B], including [A, B], [A, B, C], [A, B, X, Y], etc.
"""

from __future__ import annotations

import math
import numpy as np
import requests
from scipy import stats
from typing import Optional

# ── Ollama embed configuration ─────────────────────────────────────────────────
_EMBED_MODEL   = "all-minilm:l6-v2"
_URL_V2        = "http://localhost:11434/api/embed"       
_URL_V1        = "http://localhost:11434/api/embeddings"  
_EMBED_TIMEOUT = 10  


# ══════════════════════════════════════════════════════════════════════════════
#  Low-level helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_embedding(text: str) -> Optional[list[float]]:
    if not text or not text.strip():
        return None

    # v2 endpoint
    try:
        resp = requests.post(
            _URL_V2,
            json={"model": _EMBED_MODEL, "input": text},
            timeout=_EMBED_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        embs = data.get("embeddings") or data.get("embedding")
        if embs is not None:
            return embs[0] if isinstance(embs[0], list) else embs
    except Exception:
        pass

    # legacy endpoint
    try:
        resp = requests.post(
            _URL_V1,
            json={"model": _EMBED_MODEL, "prompt": text},
            timeout=_EMBED_TIMEOUT,
        )
        resp.raise_for_status()
        emb = resp.json().get("embedding")
        if emb is not None:
            return emb
    except Exception:
        pass

    return None

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    av, bv = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(av), np.linalg.norm(bv)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))

def _compute_n_eff(sim_matrix: list[list[float]]) -> float:
    n = len(sim_matrix)
    if n <= 1:
        return float(n)

    off_diag = [
        sim_matrix[i][j]
        for i in range(n)
        for j in range(i + 1, n)
    ]
    mean_sim = max(0.0, float(np.mean(off_diag))) if off_diag else 0.0
    return max(1.0, n / (1.0 + (n - 1) * mean_sim))

def _semantic_weights(sim_matrix: list[list[float]]) -> np.ndarray:
    n = len(sim_matrix)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([1.0])

    avg_sim = np.array([
        np.mean([sim_matrix[i][j] for j in range(n) if j != i])
        for i in range(n)
    ])
    w = np.clip(1.0 - avg_sim, 1e-9, None)
    return w / w.sum()

def _suffix_text(chain: list, prefix_len: int) -> str:
    suffix_steps = chain[prefix_len:] if len(chain) > prefix_len else chain[-1:]
    parts = [" ".join(str(seg) for seg in step) for step in suffix_steps]
    return " | ".join(parts)

def _destiny_score(
    family_fitnesses: list[float],
    all_fitnesses: list[float],
    prefix_len: int,
    avg_chain_length: float,
) -> float:
    L            = max(avg_chain_length, 1.0)
    k            = min(float(prefix_len), L)
    length_ratio = k / L

    if length_ratio >= 1.0:
        return 1.0

    if len(all_fitnesses) < 2 or len(family_fitnesses) < 2:
        return float(length_ratio) 

    n_bins   = max(5, int(math.sqrt(len(all_fitnesses))))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    def _entropy(samples: list[float]) -> float:
        counts, _ = np.histogram(samples, bins=bin_edges)
        total     = counts.sum()
        if total == 0:
            return 0.0
        probs = counts[counts > 0] / total
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    H_total  = _entropy(all_fitnesses)
    H_family = _entropy(family_fitnesses)

    mi_ratio = max(0.0, 1.0 - H_family / H_total) if H_total > 0.0 else 0.0
    D        = mi_ratio * (1.0 - length_ratio) + length_ratio
    return float(np.clip(D, 0.0, 1.0))

def _shapiro_wilk_w(scores: list[float]) -> float:
    if len(scores) < 3:
        return 0.5
    try:
        w, _ = stats.shapiro(scores)
        return float(np.clip(w, 0.0, 1.0))
    except Exception:
        return 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  LineageScorer
# ══════════════════════════════════════════════════════════════════════════════

class LineageScorer:
    def __init__(
        self,
        heritage_data: dict,
        id_to_chain: dict,
        alpha: float = 1.0,
        beta: float  = 1.0,
        n_max: float = 3.0,  # Maturity plateau threshold
        use_embeddings: bool = True,
        emb_cache: dict = None,
        verbose: bool = False,
    ) -> None:
        self.heritage_data  = heritage_data
        self.id_to_chain    = id_to_chain
        self.alpha          = alpha
        self.beta           = beta
        self.n_max          = n_max
        self.use_embeddings = use_embeddings
        self.verbose        = verbose
        self._emb_cache: dict[str, list[float]] = {}

        # USE THE PASSED CACHE, OR FALLBACK TO EMPTY DICT
        self._emb_cache: dict[str, list[float]] = emb_cache if emb_cache is not None else {}

    def _embed(self, text: str) -> Optional[list[float]]:
        if text in self._emb_cache:
            return self._emb_cache[text]
        emb = _get_embedding(text)
        if emb is not None:
            self._emb_cache[text] = emb
        return emb

    def _family_members(self, base_chain: list) -> list[dict]:
        """
        Dynamically finds all historical chains that START with `base_chain`.
        """
        chains = self.heritage_data.get("prompt_chains", {})
        members = []
        base_len = len(base_chain)
        
        for cid, record in chains.items():
            chain = self.id_to_chain.get(cid)
            if chain is None or len(chain) < base_len:
                continue
            
            # Check if historical chain shares this exact prefix
            if chain[:base_len] == base_chain:
                members.append({
                    "id":         cid,
                    "chain":      chain,
                    "fitness":    float(record.get("fitness", 0.0)),
                    "prefix_len": base_len,
                })
        return members

    def _family_raw_score(
        self,
        base_chain: list,
        all_fitnesses: list[float],
        alpha: float,
        beta: float,
    ) -> float:
        members = self._family_members(base_chain)
        if not members:
            return 0.0

        fitnesses  = [m["fitness"]    for m in members]
        chains     = [m["chain"]      for m in members]
        prefix_len = members[0]["prefix_len"]  
        n          = len(members)

        # ── Factor 3: semantic suffix diversity → n_eff & weights ─────────────
        mu: float
        sigma2: float
        n_eff: float

        if self.use_embeddings and n > 1:
            suffix_texts = [_suffix_text(c, prefix_len) for c in chains]
            embeddings   = [self._embed(t) for t in suffix_texts]

            valid_pairs  = [(emb, fitnesses[i]) for i, emb in enumerate(embeddings) if emb is not None]

            if len(valid_pairs) >= 2:
                embs = [p[0] for p in valid_pairs]
                fits = [p[1] for p in valid_pairs]
                sim_matrix = [
                    [_cosine_similarity(embs[i], embs[j]) for j in range(len(embs))]
                    for i in range(len(embs))
                ]
                n_eff  = _compute_n_eff(sim_matrix)
                weights = _semantic_weights(sim_matrix)
                mu     = float(np.average(fits, weights=weights))
                sigma2 = float(np.average([(f - mu) ** 2 for f in fits], weights=weights))
            else:
                n_eff  = float(n)
                mu     = float(np.mean(fitnesses))
                sigma2 = float(np.var(fitnesses))
        else:
            n_eff  = float(n)
            mu     = float(np.mean(fitnesses))
            sigma2 = float(np.var(fitnesses))

        # ── Factors 1, 2, 6: mean, variance, standard error ──────────────────
        sigma = math.sqrt(max(sigma2, 0.0))
        
        # CLAMP: Force a plateau so old families don't get infinite grace periods
        n_eff_clamped = min(n_eff, self.n_max)
        se = sigma / math.sqrt(max(n_eff_clamped, 1.0))

        # ── Factors 4, 5: destiny score ───────────────────────────────────────
        avg_chain_len = float(np.mean([len(c) for c in chains]))
        D = _destiny_score(fitnesses, all_fitnesses, prefix_len, avg_chain_len)

        # ── Target distribution: Shapiro–Wilk normality ───────────────────────
        W = _shapiro_wilk_w(fitnesses)

        # ── Final formula ─────────────────────────────────────────────────────
        raw = (mu - alpha * se) + beta * D * W

        if self.verbose:
            chain_preview = str(base_chain)[:30] + "..."
            print(
                f"  [LineageScorer] prefix={chain_preview:30s}"
                f"  n={n:3d}  n_eff={n_eff_clamped:5.2f}"
                f"  μ={mu:.3f}  σ={sigma:.3f}  SE={se:.3f}"
                f"  D={D:.3f}  W={W:.3f}  raw={raw:.4f}"
            )

        return float(raw)

    def _adjust_hyperparams(
        self,
        raw_scores: list[float],
    ) -> tuple[float, float]:
        if len(raw_scores) < 2:
            return self.alpha, self.beta

        score_range = float(max(raw_scores)) - float(min(raw_scores))
        if score_range < 1e-8:
            return self.alpha, self.beta

        mu_raw = abs(float(np.mean(raw_scores))) + 1e-8
        cv     = float(np.std(raw_scores)) / mu_raw

        if score_range < 0.05:          
            return self.alpha * 1.5, self.beta * 0.8
        elif cv > 1.5 or score_range > 0.8:  
            return self.alpha * 0.8, self.beta * 1.5
        else:
            return self.alpha, self.beta

    def compute_all_lineage_scores(self) -> dict[str, float]:
        chains = self.heritage_data.get("prompt_chains", {})
        if not chains:
            return {}

        all_fitnesses = [float(r.get("fitness", 0.0)) for r in chains.values()]

        if self.verbose:
            print(f"[LineageScorer] Scoring {len(chains)} unique prefix sequences")

        # ── First pass: Score every exact sequence ────────────────────────────
        family_raw: dict[str, float] = {}
        for cid, record in chains.items():
            chain = self.id_to_chain.get(cid)
            if chain:
                family_raw[cid] = self._family_raw_score(chain, all_fitnesses, self.alpha, self.beta)

        # ── Dynamic hyperparameter adjustment ─────────────────────────────────
        alpha_adj, beta_adj = self._adjust_hyperparams(list(family_raw.values()))

        if abs(alpha_adj - self.alpha) > 0.01 or abs(beta_adj - self.beta) > 0.01:
            if self.verbose:
                print(f"[LineageScorer] Hyperparams adjusted → α={alpha_adj:.3f}  β={beta_adj:.3f}")
            for cid in family_raw:
                chain = self.id_to_chain.get(cid)
                family_raw[cid] = self._family_raw_score(chain, all_fitnesses, alpha_adj, beta_adj)

        # ── Normalise to [0, 1] ───────────────────────────────────────────────
        vals    = list(family_raw.values())
        min_v   = float(min(vals))
        max_v   = float(max(vals))
        spread  = max_v - min_v

        if spread < 1e-8:
            norm_family: dict[str, float] = {cid: 0.5 for cid in family_raw}
        else:
            norm_family = {
                cid: float(np.clip((v - min_v) / spread, 0.0, 1.0))
                for cid, v in family_raw.items()
            }

        return norm_family