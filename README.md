## US Consumer Complaints — Forecasting & Anomaly Detection

This repository contains a full-stack, production-ready system that ingests the U.S. Consumer Financial Protection Bureau (CFPB) public complaint database, forecasts complaint volume for every product line, and surfaces anomalies on an interactive Streamlit dashboard, allowing a customer service team to spot spikes before they occur. The stack spans data engineering, classical and deep-learning time-series models, automated testing, CI/CD, containerization, and a one-click cloud deployment pipeline.

---

## 🔑  Key Features

| Area                  | Highlights                                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Data**              | Nightly ingestion from the CFPB Open Data API with schema evolution handling                                                    |
| **Forecasting**       | ARIMA, Facebook Prophet, and LSTM/BiLSTM/Transformer models with walk-forward validation                                        |
| **Anomaly Detection** | Dynamic thresholding (IQR + EWMA) and change-point detection (`ruptures`)                                                       |
| **Dashboard**         | Streamlit + Altair interactive UI: product picker, horizon slider, forecast vs. actual chart, anomaly flag overlay              |
| **Ops**               | Dockerfile, GitHub Actions CI, pre-commit, pytest > 90 % coverage, DVC model tracking, auto-deploy to Streamlit Community Cloud |
| **Docs**              | Sphinx site (`/docs`), in-line type hints, auto-generated API reference                                                         |
| **Roadmap**           | Slack alert bot, batch backfill job on Snowflake, and a Kafka→Spark structured-streaming path (see below)                       |

---

## 📊  Dataset

* **Source** – CFPB Consumer Complaint Database (≈ 4.9 M rows, updated daily) ([consumerfinance.gov][1])
* **Access** – pulled via the official Open Data API v1 `/complaints` endpoint with paging ([cfpb.github.io][2]).
* **Mirror** – a Kaggle snapshot enables quick offline dev and CI tests ([kaggle.com][3]).

Complaint volumes have grown nearly 5× since 2018, making forecasting and staffing increasingly important ([angle.ankura.com][4]).

---

## 🗺️  Architecture & Folder Layout

```
US-Consumer-Complaints-Forecasting/
├── data/                 # raw & processed parquet partitions (git-ignored, DVC-tracked)
├── notebooks/            # exploratory analyses & prototype models
├── src/
│   ├── ingest/           # API fetchers & delta-loader
│   ├── preprocessing/    # cleaning, time-aggregation, feature engineering
│   ├── models/           # ARIMA, Prophet, LSTM, BiLSTM, Transformer
│   ├── training/         # common training loops, CLI entry points
│   ├── evaluation/       # back-testing, metrics, visual reports
│   ├── forecasting/      # production inference wrapper
│   └── monitoring/       # anomaly detection + alert hooks
├── streamlit_app.py      # interactive dashboard (new)
├── tests/                # pytest suite (unit + integration)
├── requirements.txt      # pinned dependencies
├── Dockerfile            # reproducible runtime
└── .github/workflows/ci.yml   # lint → test → build → deploy
```

---

## 🔮  Modelling Approach

| Stage                     | Details                                                                                                                        |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Baseline**              | Seasonal ARIMA with auto-tuned `(p,d,q)(P,D,Q,s)` using `pmdarima`                                                             |
| **Classical + Exogenous** | Prophet adds holiday & linear trend components ([researchgate.net][5])                                                         |
| **Deep Learning**         | LSTM, BiLSTM and Transformer capture long-range patterns; implemented in TensorFlow/Keras with custom learning-rate schedulers |
| **Ensembling**            | Simple average and stacking regressor to reduce model dispersion                                                               |
| **Back-testing**          | Rolling-origin evaluation (expanding window) across 12-, 24-, 36-week horizons                                                 |

---

## 🚨  Anomaly Detection

We flag a data point when:

* Actual > Forecast + `k × σ` (studentised residual) **and**
* It represents an EWMA jump ≥ 2 × median absolute deviation

Change-points are detected with the `ruptures` PELT algorithm and overlaid on the chart as vertical rules.

---

## 🖥️  Interactive Dashboard

The Streamlit front-end (`streamlit_app.py`) lets analysts:

1. Choose a product (e.g., “Credit card”).
2. Toggle horizon length & model family.
3. Inspect forecast vs. actual with Altair line & area layers ([altair-viz.github.io][6]).
4. Hover to reveal anomaly metadata (date, error %, narrative count).

Run locally:

````bash
conda env create -f environment.yml   # or: pip install -r requirements.txt
streamlit run streamlit_app.py        # opens on localhost:8501
``` :contentReference[oaicite:6]{index=6}

Deployed automatically to Streamlit Community Cloud on every `main` push :contentReference[oaicite:7]{index=7}.

---

## ⚙️  Installation

```bash
git clone https://github.com/your-handle/US-Consumer-Complaints-Forecasting.git
cd US-Consumer-Complaints-Forecasting
conda create -n complaints python=3.10
conda activate complaints
pip install -r requirements.txt
````

Optional system-wide install:

```bash
pip install -e .
complaints --help
```

---

## 🐳  Docker

```
docker build -t complaints-forecast:latest .
docker run -p 8501:8501 complaints-forecast
```

The image starts the Streamlit server directly (see `ENTRYPOINT` in Dockerfile) ([docs.streamlit.io][7]).

---

## 🔁  Continuous Integration

* **GitHub Actions** – lint → tests → type-check → build → deploy.
* **Coverage** – enforced ≥ 90 % with `pytest-cov`.
* **Pre-commit** – `black`, `isort`, `ruff`, and `nbstripout` hooks.

---

## 🧪  Testing

```bash
pytest -n auto                # run unit & integration tests in parallel
pytest tests/e2e/ -m "slow"   # optional end-to-end back-test suite
```

---

## 🧭  Roadmap

| Milestone                                 | ETA     | Notes                          |
| ----------------------------------------- | ------- | ------------------------------ |
| Kafka → Spark structured-streaming ingest | Q3 2025 | Real-time anomaly alerts       |
| Prefect cloud scheduler                   | Q3 2025 | Replace cron shell scripts     |
| Slack/Teams notification bot              | Q3 2025 | Push alerts to on-call channel |
| Model registry on MLflow                  | Q4 2025 | Serve forecasts via REST       |

---

## 🤝  Contributing

Pull requests welcome! Please open an issue first to discuss major changes. Make sure `pre-commit run --all-files` passes and that your PR increases test coverage.

---

## 📄  License

This project is licensed under the **MIT License** – see [`LICENSE`](LICENSE) for details.

---

## 🙏  Acknowledgements

* CFPB for providing an open complaint database and API ([cfpb.github.io][8]).
* Streamlit & Altair teams for superb dev-experience ([docs.streamlit.io][9], [altair-viz.github.io][10]).
* Research by Vaishnav et al. on predictive analysis of CFPB data informed several modelling choices ([arxiv.org][11]).
* Community examples of complaint dashboards built on the API inspired early prototypes ([frontiergroup.org][12]).

---

### Happy forecasting!

[1]: https://www.consumerfinance.gov/data-research/consumer-complaints/?utm_source=chatgpt.com "Consumer Complaint Database"
[2]: https://cfpb.github.io/api/ccdb/?utm_source=chatgpt.com "Consumer Complaint Database API documentation - CFPB Open Tech"
[3]: https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp?utm_source=chatgpt.com "Consumer Complaints Dataset for NLP - Kaggle"
[4]: https://angle.ankura.com/post/102im68/increasing-dissatisfaction-major-areas-of-growth-in-cfpb-consumer-complaints-dat?utm_source=chatgpt.com "Major Areas of Growth in CFPB Consumer Complaints Data"
[5]: https://www.researchgate.net/publication/382111742_Predictive_Analysis_of_CFPB_Consumer_Complaints_Using_Machine_Learning?utm_source=chatgpt.com "Predictive Analysis of CFPB Consumer Complaints Using Machine ..."
[6]: https://altair-viz.github.io/altair-viz-v4/user_guide/data.html?utm_source=chatgpt.com "Specifying Data in Altair — Altair 4.2.2 documentation"
[7]: https://docs.streamlit.io/deploy/tutorials/docker?utm_source=chatgpt.com "Deploy Streamlit using Docker"
[8]: https://cfpb.github.io/api/ccdb/api.html?utm_source=chatgpt.com "Consumer Complaint Database API documentation - CFPB Open Tech"
[9]: https://docs.streamlit.io/develop/concepts/architecture/run-your-app?utm_source=chatgpt.com "Run your Streamlit app"
[10]: https://altair-viz.github.io/user_guide/display_frontends.html?utm_source=chatgpt.com "Displaying Altair Charts — Vega-Altair 5.5.0 documentation"
[11]: https://arxiv.org/abs/2407.06399?utm_source=chatgpt.com "Predictive Analysis of CFPB Consumer Complaints Using Machine Learning"
[12]: https://frontiergroup.org/articles/how-explore-consumer-problems-financial-marketplace-using-cfpbs-consumer-complaint/?utm_source=chatgpt.com "How to explore consumer problems in the financial marketplace ..."
