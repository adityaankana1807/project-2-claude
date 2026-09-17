"""Project-wide constants for the India spatio-temporal crime forecasting pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SYNTHETIC = ROOT / "data" / "synthetic"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

for d in [DATA_PROCESSED, DATA_SYNTHETIC, MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# The 7 NCRB "Crime Against Women" categories present in the real 2001 district file.
CRIME_CATEGORIES = [
    "Rape",
    "Kidnapping_&_Abduction",
    "Dowry_Deaths",
    "Assault_on_women_with_intent_to_outrage_her_modesty",
    "Insult_to_modesty_of_Women",
    "Cruelty_by_Husband_or_his_Relatives",
    "Importation_of_Girls",
]

# Simulation horizon for the synthetic panel.
SIM_START_YEAR = 2001
SIM_END_YEAR = 2024

# Spatial graph: number of nearest-neighbour districts each district is connected to.
KNN_NEIGHBORS = 6

# Near-repeat / self-excitation (Hawkes-style) kernel applied on top of the
# seasonal + population + trend baseline, motivated by the modus-operandi /
# near-repeat burglary literature and spatio-temporal point-process papers.
NEAR_REPEAT_DECAY_MONTHS = 2.0     # exponential decay time constant, in months
NEAR_REPEAT_SELF_WEIGHT = 0.35     # excitation contributed to the same district
NEAR_REPEAT_NEIGHBOR_WEIGHT = 0.12  # excitation contributed to each graph neighbour

# Illustrative monthly seasonal multipliers per category (not fitted from a
# ground-truth monthly series, since NCRB only publishes annual district
# totals -- these encode documented criminological patterns: a wedding /
# festive season effect (Oct-Feb) for dowry & cruelty-by-husband categories,
# and a summer-holiday mobility effect (Apr-Jun) for stranger-perpetrated
# categories such as rape / kidnapping / assault. They are clearly labelled
# as modelling assumptions in the README and are a configuration point, not
# a claimed empirical finding.
SEASONAL_WEIGHTS = {
    "Rape": [1.02, 0.97, 1.03, 1.08, 1.10, 1.05, 0.95, 0.92, 0.95, 1.00, 0.98, 0.95],
    "Kidnapping_&_Abduction": [1.00, 0.95, 1.02, 1.10, 1.15, 1.10, 0.98, 0.92, 0.92, 0.98, 0.95, 0.93],
    "Dowry_Deaths": [1.10, 1.05, 0.98, 0.90, 0.88, 0.90, 0.92, 0.95, 1.00, 1.08, 1.15, 1.12],
    "Assault_on_women_with_intent_to_outrage_her_modesty": [1.00, 0.96, 1.02, 1.08, 1.12, 1.06, 0.96, 0.92, 0.94, 1.00, 1.00, 0.98],
    "Insult_to_modesty_of_Women": [1.00, 0.97, 1.01, 1.06, 1.08, 1.04, 0.97, 0.94, 0.96, 1.00, 1.00, 0.99],
    "Cruelty_by_Husband_or_his_Relatives": [1.08, 1.03, 0.98, 0.92, 0.90, 0.92, 0.94, 0.97, 1.00, 1.06, 1.12, 1.10],
    "Importation_of_Girls": [1.00, 0.98, 1.00, 1.04, 1.06, 1.04, 0.98, 0.96, 0.98, 1.00, 1.00, 0.98],
}

# Damped annual growth rate applied once real (2001-2012) national trends run out.
EXTRAPOLATION_GROWTH_CAP = 0.06
POPULATION_ANNUAL_GROWTH = 0.012
