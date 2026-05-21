# survRM2py

[![PyPI version](https://img.shields.io/pypi/v/survrm2py.svg)](https://pypi.org/project/survrm2py/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

`survRM2py` is a Python reimplementation of the popular R package [survRM2](https://cran.r-project.org/web/packages/survRM2/index.html) for Restricted Mean Survival Time (RMST) analysis. It is designed to achieve absolute mathematical parity with the R reference implementation.

RMST is a robust alternative to the hazard ratio for comparing survival curves, representing the average survival time up to a pre-specified time point (tau).

---

## Installation

### Standard Python Environment
You can install the package directly from PyPI:
```bash
pip install survrm2py
```

---

## Quick Start Example

```python
import numpy as np
import pandas as pd
from survrm2py import run_rmst

# 1. Create mock survival data
np.random.seed(42)
n = 200
data = pd.DataFrame({
    "time": np.random.exponential(scale=10, size=n),
    "event": np.random.binomial(n=1, p=0.7, size=n),
    "arm": np.random.binomial(n=1, p=0.5, size=n),
    "age": np.random.normal(loc=60, scale=10, size=n)
})

# 2. Run unadjusted and covariate-adjusted RMST at tau = 8.0
results = run_rmst(
    df=data,
    time_col="time",
    event_col="event",
    arm_col="arm",
    tau=8.0,
    covariates=["age"],
    method="ipcw_rmst2"
)

# 3. Print unadjusted differences
print(f"RMST Arm 1: {results['rmst_arm1']:.4f}")
print(f"RMST Arm 0: {results['rmst_arm0']:.4f}")
print(f"Unadjusted Diff: {results['rmst_diff_unadjusted']:.4f} (p={results['p_unadjusted']:.4f})")

# 4. View adjusted covariate summary table
print("\nAdjusted IPCW Regression Summary:")
print(results["adjusted_summary"])
```

---

## Developer Guide (Pixi)

This package uses a unified `pyproject.toml` with `tool.pixi` environment manager.

### Prerequisites
Make sure you have [Pixi](https://pixi.sh/) installed:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Running Tests
The test environment automatically sets up Python, R, and installs the reference `survRM2` package from CRAN.
```bash
pixi run -e test test
```

### Building Package Distribution Files
To build the distribution packages (`.whl` and `.tar.gz`) locally in a clean sandboxed environment:
```bash
pixi run -e test build
```
Built files will be generated under the `dist/` directory.

---

## License

This package is licensed under the **GNU General Public License v3 or later (GPLv3+)**, which maintains consistency with CRAN's `survRM2` licensing.

## References
Uno H, Claggett B, Tian L, Inoue E, Gallo P, Miyata T, Schrag D, Takeuchi M, Uyama Y, Zhao L, Skali H, Solomon S, Jacobus S, HughesM, Packer M, Wei LJ. Moving beyond the hazard ratio in quantifying the between-group difference in survival analysis. Journal of clinical Oncology 2014, 32, 2380-2385. doi:10.1200/JCO.2014.55.2208.

Tian L, Zhao L, Wei LJ. Predicting the restricted mean event time with the subject's baseline covariates in survival analysis. Biostatistics 2014, 15, 222-233. doi:10.1093/biostatistics/kxt050.