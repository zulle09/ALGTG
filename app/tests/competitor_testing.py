"""Helpers shared by the competitor test files (test_typiclust.py, test_probcover.py, ...)."""

import contextlib
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ActiveLearner import ActiveLearner
from competitor_config import competitor_params


@contextlib.contextmanager
def config(strategy: str, **overrides):
    """Temporarily override entries of competitor_params[strategy]; strategies copy them at construction."""
    saved = dict(competitor_params[strategy])
    competitor_params[strategy].update(overrides)
    try:
        yield
    finally:
        competitor_params[strategy].clear()
        competitor_params[strategy].update(saved)


def build_strategy(strategy_class, dataset_name='cifar10', **ct_p):
    """A real strategy object, constructed without the heavy ActiveLearner.__init__ (dataset, model, folders)."""
    def light_init(self, ct_p, t_p, method_name):
        self.ct_p, self.strategy_name = ct_p, f'ResNet_{method_name}'

    with mock.patch.object(ActiveLearner, '__init__', light_init):
        return strategy_class({'dataset_name': dataset_name, 'trial': 0, **ct_p}, {})
