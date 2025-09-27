your-project/
├─ README.md
├─ LICENSE
├─ pyproject.toml
├─ .pre-commit-config.yaml
├─ .gitattributes
├─ notebooks/
│  ├─ 00_exploration.ipynb
│  ├─ 10_data_checks.ipynb
│  └─ 90_results_review.ipynb
├─ configs/
│  ├─ data.yaml
│  ├─ env.yaml
│  ├─ rl.yaml
│  └─ regimes.yaml
├─ data/
│  ├─ sample/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ src/
│  ├─ __init__.py
│  ├─ data_pipeline/
│  │  ├─ ingest.py
│  │  ├─ clean.py
│  │  └─ features.py
│  ├─ regimes/
│  │  ├─ unsupervised.py
│  │  ├─ hmm.py
│  │  └─ labeling.py
│  ├─ env/
│  │  ├─ trading_env.py
│  │  ├─ portfolio.py
│  │  └─ render_utils.py
│  ├─ agents/
│  │  ├─ baselines.py
│  │  ├─ dqn.py
│  │  └─ ppo.py
│  ├─ evaluation/
│  │  ├─ backtest.py
│  │  ├─ metrics.py
│  │  └─ plots.py
│  └─ utils/
│     ├─ io.py
│     ├─ logging.py
│     └─ seeding.py
├─ experiments/
│  ├─ exp001_baselines.yaml
│  ├─ exp002_regime_kmeans.yaml
│  └─ exp010_dqn_v1.yaml
├─ scripts/
│  ├─ run_data_pipeline.py
│  ├─ run_regime_fit.py
│  ├─ run_backtest.py
│  └─ run_sweep.py
├─ results/
│  ├─ logs/
│  ├─ plots/
│  └─ artifacts/
└─ tests/
   ├─ test_data_pipeline.py
   ├─ test_env.py
   ├─ test_portfolio.py
   └─ test_metrics.py
