Code for  CAP5150 final report comparing coarsening, suppression, and a differentially private Laplace mechanism against a database reconstruction attack. All data is synthetic.

## Files

- run_pilot.py: generates the synthetic data, creates the protected tables, and runs reconstruction
- step1_ties_baseline.py: adds tie handling and baseline comparisons
- results.json, summary.csv, releases.json, original_records.csv: outputs of run_pilot.py
- step1_results.json: output of step1_ties_baseline.py

## How to run

Requires Python 3 with NumPy and SciPy (1.9 or later).

```
pip install numpy scipy
python run_pilot.py
python step1_ties_baseline.py
```
