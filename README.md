# HPO Engine: A Professional Hyperparameter Optimization Library

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)](https://github.com/jules-agent/hpo-engine/actions)

**HPO Engine** is a lightweight, powerful, and easy-to-use Python library for hyperparameter optimization. It is designed to help machine learning engineers and researchers efficiently find the best hyperparameters for their models using modern optimization algorithms like the Tree-structured Parzen Estimator (TPE).

This project refactors an initial prototype into a professional, modular, and well-tested library suitable for production environments.

## ✨ Key Features

- **Modern Optimization Algorithms**: Uses TPE for efficient, intelligent search.
- **Simple and Flexible API**: Define complex search spaces with just a few lines of code.
- **Extensible by Design**: Easily add custom samplers or pruners.
- **Built-in Visualization**: Automatically generate plots to understand the optimization process.
- **Lightweight and Minimal Dependencies**: Built on top of standard scientific libraries like NumPy and SciPy.
- **Asynchronous Pruning (Coming Soon)**: Includes implementations of Median and ASHA pruners for early-stopping of unpromising trials.

## 🚀 Quick Start

### 1. Installation

First, clone the repository and install the required dependencies. It is recommended to do this in a virtual environment.

```bash
# Clone the repository
git clone https://github.com/jules-agent/hpo-engine.git
cd hpo-engine

# Install core dependencies
pip install -r requirements.txt

# Install the HPO engine in editable mode
pip install -e .
```

### 2. Run an Example

The `examples` directory contains scripts that demonstrate how to use the library.

**To run the basic mathematical optimization example:**

```bash
python run_hpo.py math
```

This will run `examples/01_mathematical_optimization.py`, which finds the maximum of a simple 2D function. It will print the results to the console and save a visualization of the optimization history as `mathematical_optimization.png`.

![Mathematical Optimization History](https://raw.githubusercontent.com/jules-agent/hpo-engine/main/mathematical_optimization.png)

**To run the Scikit-learn optimization example:**

```bash
python run_hpo.py sklearn
```

This will run `examples/02_sklearn_optimization.py`, which tunes the hyperparameters of a `RandomForestClassifier`. It will save its plot as `sklearn_optimization.png`.

## ⚙️ How It Works

Using the HPO Engine involves three main steps:

**1. Define an Objective Function:**
This is a Python function that takes a `trial` object, uses it to get hyperparameters, and returns a score to be maximized or minimized.

```python
def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    return -(x - 2)**2 - (y + 2)**2
```

**2. Define the Search Space:**
Create a `SearchSpace` object and add the hyperparameters you want to tune.

```python
from hpo import SearchSpace

search_space = SearchSpace()
search_space.add_uniform('x', -10, 10)
search_space.add_uniform('y', -10, 10)
```

**3. Create and Run a Study:**
A `Study` orchestrates the optimization. You provide it with the search space and objective function, and it handles the rest.

```python
from hpo import Study

study = Study(
    search_space=search_space,
    objective_function=objective,
    direction='maximize',  # or 'minimize'
    n_trials=100
)

best_trial = study.optimize()

print(f"Best value: {best_trial.value}")
print(f"Best params: {best_trial.params}")
```

## ✅ Running Tests

The project includes a comprehensive test suite. To run the tests, first install the development dependencies:

```bash
pip install -r dev-requirements.txt
```

Then, run `pytest`:

```bash
pytest
```

This will execute all tests and provide a coverage report.