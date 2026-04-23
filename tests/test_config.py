from mac.config import load_config


def test_default_config_loads():
    cfg = load_config()
    assert "analyst" in cfg.agents
    assert "programmer" in cfg.agents
    assert "reviewer" in cfg.agents
    assert cfg.budgets.max_iterations >= 1
    # every agent references a defined model
    for name, a in cfg.agents.items():
        assert a.model in cfg.models, f"{name}: unknown model {a.model}"
        for fb in a.fallback:
            assert fb in cfg.models, f"{name}: unknown fallback {fb}"
