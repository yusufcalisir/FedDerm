# FedDerm Tests

Tests live in this directory and are run with:

```bash
pytest tests/ -v
```

## Test categories

| File | Phase | Purpose |
|------|-------|---------|
| `test_environment.py` | Phase 1 | Dependency imports + device availability |
| `test_data.py` *(Phase 2)* | Phase 2 | DermaMNIST loading, split shapes |
| `test_partitioning.py` *(Phase 3)* | Phase 3 | Dirichlet split reproducibility |
| `test_strategies.py` *(Phase 3)* | Phase 3 | FedAvg / FedProx round logic |
| `test_privacy.py` *(Phase 4)* | Phase 4 | DP accountant ε/δ accounting |
