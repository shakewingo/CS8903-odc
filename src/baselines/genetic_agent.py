"""Genetic Algorithm baseline.

Chromosome: fixed-length (= env.max_steps) integer sequence of flat action
indices. Fitness: final `total_value − initial total_value` when the chromosome
is replayed against a snapshot of the env positioned at the target grid.

Invalid genes (masked-out at replay time) are no-ops; during evolution and
replay, `max_consecutive_noops` is raised above the horizon so the chromosome
is always evaluated over its full length.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from src.baselines.common import reset_to_index


class GeneticAgent:
    def __init__(
        self,
        pop_size: int = 30,
        generations: int = 50,
        tournament_k: int = 3,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.02,
        n_elite: int = 2,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.pop_size = pop_size
        self.generations = generations
        self.tournament_k = tournament_k
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.n_elite = n_elite
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self._action_dim: Optional[int] = None

    # ── Main solve loop ───────────────────────────────────────────────
    def solve(self, env, idx: int) -> Callable:
        """Evolve a best-fitness chromosome for the sample at `env.samples[idx]`.

        Returns `act_fn(env) -> int` that replays chromosome gene
        `chromosome[env.step_count]` each step. Leaves env at the initial
        snapshot so the caller can run the replay episode directly.
        """
        horizon = env.max_steps
        high_noops = horizon + 1
        self._action_dim = int(env.action_dim)

        reset_to_index(env, idx, max_consecutive_noops_override=high_noops)
        initial_snapshot = env.get_state_dict()
        initial_total = env.prev_total_value

        # Mask-aware seeding: each chromosome gene starts legal under the initial
        # action mask, avoiding a pure-no-op starting population.
        init_mask_valid = np.flatnonzero(env.action_masks())
        pop = [
            self.rng.choice(init_mask_valid, size=horizon).astype(np.int64)
            for _ in range(self.pop_size)
        ]

        def fitness(chromosome: np.ndarray) -> float:
            env.set_state_dict(initial_snapshot)
            for gene in chromosome:
                _, _, terminated, _, _ = env.step(int(gene))
                if terminated:
                    break
            return env.prev_total_value - initial_total

        scores = np.array([fitness(c) for c in pop], dtype=np.float64)

        for gen in range(self.generations):
            elite_idx = np.argsort(scores)[-self.n_elite:]
            elites = [pop[i].copy() for i in elite_idx]

            new_pop: list[np.ndarray] = list(elites)
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, scores)
                p2 = self._tournament_select(pop, scores)
                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1)
                self._mutate(c2)
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            pop = new_pop
            scores = np.array([fitness(c) for c in pop], dtype=np.float64)

            if self.verbose:
                print(f"  gen {gen+1}/{self.generations}: "
                      f"best={scores.max():.4f} mean={scores.mean():.4f} "
                      f"std={scores.std():.4f}")

        best_idx = int(np.argmax(scores))
        best_chromosome = pop[best_idx]

        # Leave env in a deterministic state; caller will reset_to_index again.
        env.set_state_dict(initial_snapshot)

        def act_fn(env) -> int:
            return int(best_chromosome[env.step_count])
        act_fn.chromosome = best_chromosome
        return act_fn

    # ── Operators ─────────────────────────────────────────────────────
    def _tournament_select(self, pop: list[np.ndarray], scores: np.ndarray) -> np.ndarray:
        k = min(self.tournament_k, len(pop))
        contenders = self.rng.choice(len(pop), size=k, replace=False)
        winner = contenders[int(np.argmax(scores[contenders]))]
        return pop[winner].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.rng.random() >= self.crossover_prob:
            return p1.copy(), p2.copy()
        point = int(self.rng.integers(1, len(p1)))
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
        return c1, c2

    def _mutate(self, chromosome: np.ndarray) -> None:
        """In-place mutation: flip each gene with prob `mutation_prob`."""
        mut_mask = self.rng.random(len(chromosome)) < self.mutation_prob
        n_mut = int(mut_mask.sum())
        if n_mut == 0:
            return
        chromosome[mut_mask] = self.rng.integers(0, self._action_dim, size=n_mut)
