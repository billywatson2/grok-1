#!/usr/bin/env python3
"""Checks for the rating math in arena.py.

    python3 test_ratings.py

Pure functions only -- no models, no server, no network. Run this after touching
`elo_update`, `bradley_terry` or `bootstrap_ci`.
"""

from __future__ import annotations

import unittest

from arena import (ELO_K, ELO_START, bootstrap_ci, bradley_terry, elo_update,
                   expected_score)


class TestElo(unittest.TestCase):
    def test_symmetry(self):
        """A win and a loss must mirror each other exactly."""
        win_a, win_b = elo_update(1000, 1000, 1.0)
        loss_a, loss_b = elo_update(1000, 1000, 0.0)
        self.assertAlmostEqual(win_a - 1000, ELO_K / 2)
        self.assertAlmostEqual(loss_a - 1000, -ELO_K / 2)
        self.assertAlmostEqual(win_a - 1000, -(loss_a - 1000))
        self.assertAlmostEqual(win_b - 1000, -(loss_b - 1000))

    def test_zero_sum(self):
        for score in (0.0, 0.5, 1.0):
            a, b = elo_update(1234.0, 987.0, score)
            self.assertAlmostEqual((a - 1234.0) + (b - 987.0), 0.0)

    def test_tie_is_small_and_opposite(self):
        a, b = elo_update(1000, 1000, 0.5)
        self.assertAlmostEqual(a, 1000.0)
        self.assertAlmostEqual(b, 1000.0)
        # a tie between unequal ratings still moves both a little
        a, b = elo_update(1200, 1000, 0.5)
        self.assertGreater(b - 1000, 0)
        self.assertLess(a - 1200, 0)

    def test_favourite_gains_less(self):
        strong_win, _ = elo_update(1300, 1000, 1.0)
        underdog_win, _ = elo_update(1000, 1300, 1.0)
        self.assertGreater(underdog_win - 1000, strong_win - 1300)

    def test_expected_bounds(self):
        self.assertAlmostEqual(expected_score(1000, 1000), 0.5)
        self.assertAlmostEqual(expected_score(1000, 1400), 0.0909, places=3)


class TestBradleyTerry(unittest.TestCase):
    def test_dominant_model_ranks_higher(self):
        # model 1 beats model 2 five times, loses once
        battles = [(1, 2, 1.0)] * 5 + [(2, 1, 1.0)]
        fit = bradley_terry(battles, [1, 2])
        self.assertGreater(fit[1], fit[2])
        # symmetric case -> equal ratings
        fit_even = bradley_terry([(1, 2, 1.0), (2, 1, 1.0)], [1, 2])
        self.assertAlmostEqual(fit_even[1], fit_even[2], places=6)

    def test_ties_pull_toward_each_other(self):
        decisive = bradley_terry([(1, 2, 1.0)], [1, 2])
        tied = bradley_terry([(1, 2, 0.5)], [1, 2])
        self.assertGreater(decisive[1] - decisive[2], tied[1] - tied[2])

    def test_scale_is_elo_like(self):
        fit = bradley_terry([(1, 2, 1.0)] * 10, [1, 2])
        self.assertGreater(fit[1], 1100)
        self.assertLess(fit[2], 900)

    def test_no_battles_gives_start_rating(self):
        fit = bradley_terry([], [1, 2])
        self.assertEqual(fit, {1: ELO_START, 2: ELO_START})


class TestBootstrap(unittest.TestCase):
    def test_interval_brackets_point_estimate(self):
        battles = [(1, 2, 1.0)] * 6 + [(2, 1, 1.0)] * 4
        ids = [1, 2]
        ci = bootstrap_ci(battles, ids, samples=60)
        point = bradley_terry(battles, ids)
        for i in ids:
            lo, hi = ci[i]
            self.assertLessEqual(lo, point[i] + 1e-6)
            self.assertGreaterEqual(hi, point[i] - 1e-6)
            self.assertLessEqual(hi - lo, 800)

    def test_no_battles(self):
        ci = bootstrap_ci([], [1, 2], samples=10)
        self.assertEqual(ci[1], (ELO_START, ELO_START))


if __name__ == "__main__":
    unittest.main(verbosity=2)
