# Partnership ranking methods

# Implements ranking logic and scoring features like overlap and cost savings.
# Validates rankings using heuristics and exact VRP results.

# Role: Implements ranking logic.
# Interactions:
# With dss.py:
# Provides ranking algorithms and outputs scores based on features like overlap and savings potential.
# Ensures rankings are consistent with DSS results by validating with small VRP solutions.
# With routing.py:
# Compares rankings to exact VRP solutions for validation (top and bottom-ranked partnerships).

import pandas as pd

def get_mock_ranking():
    """
    Returns a mock ranking table for collaborations.
    """
    mock_data = pd.DataFrame({
        "Rank": [1, 2, 3],
        "Company A": ["Company 1", "Company 2", "Company 3"],
        "Company B": ["Company 4", "Company 5", "Company 6"],
        "Savings (€)": [250.00, 200.50, 150.75]
    })
    return mock_data