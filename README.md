# HAL Defence Manufacturing – Operations Research Dashboard

Streamlit dashboard with seven models solved live in Python (SciPy/HiGHS):
LP, Goal Programming, Integer Programming, Transportation, Assignment, Nonlinear Programming, TSP.
All inputs are academic planning assumptions, not HAL internal data.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit interface (tabs, charts, interactive inputs) |
| `models.py` | Data and solver functions for all seven models |
| `requirements.txt` | Python packages Streamlit Cloud installs |
| `.streamlit/config.toml` | Theme / server settings |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free, public link)
1. Create a **public** GitHub repository (e.g. `operationsmanagementdb`) and upload all files, keeping the
   `.streamlit/` folder. `app.py` must be at the repository root.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** -> **Deploy a public app from GitHub**.
4. Choose your repository, branch `main`, main file path `app.py`.
5. Open **Advanced settings**, pick Python 3.11 or 3.12, and set the custom subdomain, e.g. `operationsmanagementdb`.
6. Click **Deploy**. Your link will be `https://operationsmanagementdb.streamlit.app` (anyone can open it).

If the app is ever set to private, use the app's **Share** menu and choose "This app is public".
