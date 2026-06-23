"""
Genetic_algorithm_processes/S2_recombination/methods/pairing/lineage_pairing.py

3-mode lineage-aware pairing for prompt chain recombination.

Conceptual model
────────────────
Every pairing produces (P1, P2, crossover_hint, mode):

  P1 = prefix donor  → its family_id is inherited by the offspring
  P2 = suffix donor  → contributes steps after the crossover point

  crossover_hint = None   → NCrossoverMultiparent uses its random distribution
                = int k  → crossover fixed at step k
                            (k = len(P1) means "take all of P1 as prefix")
                            
  mode = string → the exact recombination strategy used ("exploitation", "infidelity", "mentorship")

Mode probabilities (normalised from constructor args)
─────────────────────────────────────────────────────
  1. Standard Exploitation  (default 25 %)
     High-Lf P1 + same-family P2 (like simulated annealing on the suffix).
     Crossover: random (explores suffix combinations within the family).

  2. Infidelity / Exploration  (default 45 %)
     Low-Lf P1 (new/unstable family) + High-Lf P2 (proven suffix donor).
     Crossover fixed at len(P1): take all of P1 as the prefix, graft P2's
     continuation.

  3. Mentorship / Exploration  (default 30 %)
     High-fitness + Low-Lf P1 ("Godfather") + global best-lineage P2.
     Crossover fixed at len(P1): Godfather's chain becomes the new prefix.

Input population format (from GeneralDataManager.get_population_with_lineage)
──────────────────────────────────────────────────────────────────────────────
  list of (chain, fitness, lineage_score, family_id)
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


# ── Named pair type ────────────────────────────────────────────────────────────
# (p1_chain, p2_chain, crossover_hint, mode)
Pair = tuple[list, list, Optional[int], str]


class LineagePairing:
    """
    Produces pairs of (P1, P2, crossover_hint, mode) using one of three lineage-
    driven strategies chosen stochastically each round.
    """

    def __init__(
        self,
        exploitation_ratio:            float = 0.25,
        infidelity_ratio:              float = 0.45,
        mentorship_ratio:              float = 0.30,
        mentorship_fitness_percentile: float = 0.90,
        mentorship_lineage_threshold:  float = 0.30,
        number_of_pairs:               Optional[int] = None,
        verbose:                       bool  = False,
    ) -> None:
        total = exploitation_ratio + infidelity_ratio + mentorship_ratio
        self.exploitation_ratio            = exploitation_ratio / total
        self.infidelity_ratio              = infidelity_ratio   / total
        self.mentorship_ratio              = mentorship_ratio   / total
        self.mentorship_fitness_percentile = mentorship_fitness_percentile
        self.mentorship_lineage_threshold  = mentorship_lineage_threshold
        self.number_of_pairs               = number_of_pairs
        self.verbose                       = verbose

        # Cumulative thresholds for fast mode selection
        self._thresholds = np.cumsum([
            self.exploitation_ratio,
            self.infidelity_ratio,
        ])

    # ── Internal sampling ─────────────────────────────────────────────────────

    @staticmethod
    def _weighted_choice(
        candidates: list,
        weights:    np.ndarray,
        exclude_id: Optional[int] = None,
    ):
        if exclude_id is not None:
            mask       = np.array([id(c) != exclude_id for c in candidates])
            candidates = [c for c, m in zip(candidates, mask) if m]
            weights    = weights[mask]

        if len(candidates) == 0:
            return None

        weights = np.clip(weights, 0.0, None)
        total   = weights.sum()
        if total <= 0.0:
            weights = np.ones(len(candidates))
            total   = float(len(candidates))

        probs = weights / total
        idx   = np.random.choice(len(candidates), p=probs)
        return candidates[idx]

    # ── Mode implementations ──────────────────────────────────────────────────

    def _exploitation_pair(
        self,
        pop:              list,  
        high_lf_cutoff:   float = 0.50,
    ) -> Optional[Pair]:
        chains, fitnesses, lf_scores, family_ids = zip(*pop)
        lf_arr = np.asarray(lf_scores, float)

        high_mask = lf_arr >= max(high_lf_cutoff, np.median(lf_arr))
        if not high_mask.any():
            high_mask = np.ones(len(pop), dtype=bool)

        p1_cands   = [pop[i] for i in range(len(pop)) if high_mask[i]]
        p1_weights = lf_arr[high_mask]
        p1_entry   = self._weighted_choice(p1_cands, p1_weights)
        if p1_entry is None:
            return None
        p1_chain, _, _, p1_fid = p1_entry

        same_fam = [e for e in pop if e[0][0] == p1_chain[0] and id(e[0]) != id(p1_chain)]
        if same_fam:
            sf_lf = np.asarray([e[2] for e in same_fam], float)
            p2_entry = self._weighted_choice(same_fam, sf_lf)
        else:
            p2_entry = self._weighted_choice(p1_cands, p1_weights, exclude_id=id(p1_chain))

        if p2_entry is None:
            return None

        return (p1_chain, p2_entry[0], None, "exploitation")

    def _infidelity_pair(
        self,
        pop:             list,
        low_lf_cutoff:   float = 0.50,
    ) -> Optional[Pair]:
        chains, fitnesses, lf_scores, family_ids = zip(*pop)
        lf_arr = np.asarray(lf_scores, float)

        low_mask  = lf_arr <= min(low_lf_cutoff, np.median(lf_arr))
        if not low_mask.any():
            low_mask = np.ones(len(pop), dtype=bool)

        p1_cands   = [pop[i] for i in range(len(pop)) if low_mask[i]]
        inv_weights = np.clip(1.0 - lf_arr[low_mask], 1e-9, None)
        p1_entry    = self._weighted_choice(p1_cands, inv_weights)
        if p1_entry is None:
            return None
        p1_chain, _, _, p1_fid = p1_entry

        diff_fam = [e for e in pop if e[0][0] != p1_chain[0]]
        if not diff_fam:
            diff_fam = [e for e in pop if id(e[0]) != id(p1_chain)]
        if not diff_fam:
            return None

        df_lf    = np.asarray([e[2] for e in diff_fam], float)
        p2_entry = self._weighted_choice(diff_fam, df_lf)
        if p2_entry is None:
            return None

        crossover_hint = max(1, len(p1_chain))
        return (p1_chain, p2_entry[0], crossover_hint, "infidelity")

    def _mentorship_pair(
        self,
        pop: list,
    ) -> Optional[Pair]:
        _, fitnesses, lf_scores, _ = zip(*pop)
        fit_arr = np.asarray(fitnesses, float)
        lf_arr  = np.asarray(lf_scores,  float)

        fit_threshold = float(np.percentile(fit_arr, self.mentorship_fitness_percentile * 100))

        godfather_mask = (fit_arr >= fit_threshold) & (lf_arr < self.mentorship_lineage_threshold)

        if not godfather_mask.any():
            n_top          = max(1, len(pop) // 5)
            top_indices    = np.argsort(fit_arr)[-n_top:]
            godfather_mask = np.zeros(len(pop), dtype=bool)
            godfather_mask[top_indices] = True

        gf_cands   = [pop[i] for i in range(len(pop)) if godfather_mask[i]]
        gf_weights = fit_arr[godfather_mask]
        p1_entry   = self._weighted_choice(gf_cands, gf_weights)
        if p1_entry is None:
            return None
        p1_chain, _, _, _ = p1_entry

        role_scores = fit_arr * lf_arr
        p2_entry    = self._weighted_choice(pop, role_scores, exclude_id=id(p1_chain))
        if p2_entry is None:
            return None

        crossover_hint = max(1, len(p1_chain))
        return (p1_chain, p2_entry[0], crossover_hint, "mentorship")

    # ── Public API ────────────────────────────────────────────────────────────

    def pair(
        self,
        population_with_lineage: list,
    ) -> list[Pair]:
        pop     = population_with_lineage
        n_pairs = self.number_of_pairs or len(pop)

        if len(pop) < 2:
            return []

        pairs: list[Pair] = []
        mode_counts = {"exploitation": 0, "infidelity": 0, "mentorship": 0}

        for _ in range(n_pairs):
            r = random.random()

            if r < self._thresholds[0]:
                pair = self._exploitation_pair(pop)
                if pair: mode_counts["exploitation"] += 1
            elif r < self._thresholds[1]:
                pair = self._infidelity_pair(pop)
                if pair: mode_counts["infidelity"] += 1
            else:
                pair = self._mentorship_pair(pop)
                if pair: mode_counts["mentorship"] += 1

            if pair is not None:
                pairs.append(pair)

        if self.verbose:
            total = sum(mode_counts.values()) or 1
            print(
                f"[LineagePairing] {len(pairs)} pairs generated  |  "
                f"Exploitation {mode_counts['exploitation']:3d} ({mode_counts['exploitation']/total:.0%})  "
                f"Infidelity {mode_counts['infidelity']:3d} ({mode_counts['infidelity']/total:.0%})  "
                f"Mentorship {mode_counts['mentorship']:3d} ({mode_counts['mentorship']/total:.0%})"
            )

        return pairs


# ══════════════════════════════════════════════════════════════════════════════
#  Quick smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Synthetic population: (chain, fitness, lineage_score, family_id)
    pop_sim = [
        ([("gpt-4", "Summarize the question.")],              0.85, 0.80, "fam_A"),
        ([("gpt-4", "Summarize the question."), ("llm", "Answer concisely.")], 0.72, 0.75, "fam_A"),
        ([("llama3", "Rephrase the question.")],              0.40, 0.15, "fam_B"),
        ([("llama3", "Rephrase the question."), ("gpt", "Now answer.")],       0.55, 0.20, "fam_B"),
        ([("mistral", "Think step by step.")],                0.90, 0.05, "fam_C"),  
        ([("mistral", "Think step by step."), ("phi", "Summarise.")],         0.65, 0.60, "fam_D"),
    ]

    pairer = LineagePairing(verbose=True, number_of_pairs=10)
    pairs  = pairer.pair(pop_sim)

    print(f"\nGenerated {len(pairs)} pairs:")
    for i, (p1, p2, hint, mode) in enumerate(pairs):
        print(f"  [{i+1}] Mode={mode:12s} P1 steps={len(p1)}  P2 steps={len(p2)}  crossover_hint={hint}")