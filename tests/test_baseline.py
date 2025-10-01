# tests/test_baselines.py

import pytest
import numpy as np
from src.agents.baselines import EMABaselineAgent


def test_agent_initialization():
    agent = EMABaselineAgent(short_window=5, long_window=20)
    assert agent.short_window == 5
    assert agent.long_window == 20
    assert agent.last_action == 0
    assert agent.prices == []

def test_invalid_windows():
    with pytest.raises(ValueError):
        EMABaselineAgent(short_window=20, long_window=10)

def test_reset_clears_state():
    agent = EMABaselineAgent(5, 20)
    agent.prices = [1, 2, 3]
    agent.last_action = 1
    agent.reset()
    assert agent.prices == []
    assert agent.last_action == 0

def test_act_outputs_valid_actions():
    agent = EMABaselineAgent(3, 5)

    # feed in a rising series of prices
    prices = np.arange(1, 20)

    actions = [agent.act(p) for p in prices]

    # all actions must be in {-1, 0, 1}
    assert all(a in (-1, 0, 1) for a in actions)

    # after enough data, agent should eventually issue buy (1)
    assert 1 in actions
