# NLP1-Natural_Language_Inference-Project

## Run the data pipeline

From the project root, using your NLP Python environment:

```powershell
python run_data.py
```

For a new environment:

```powershell
uv sync
uv run python run_data.py
```

Or run this in your NLP notebook kernel from `notebooks/`:

```python
%run ../run_data.py
```

Settings are in `run_data.py`. The default is 50,000 examples; use 1,000 for a quick check.
