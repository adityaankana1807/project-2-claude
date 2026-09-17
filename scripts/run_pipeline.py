"""One-command end-to-end pipeline:
harmonize real data -> build spatial graph -> synthesize district-month
panel -> train ST-SAGE -> evaluate -> run the crime-linkage demo.

Usage:  .venv/Scripts/python.exe scripts/run_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import harmonize, spatial_graph, synthesize
from src.linkage import run_linkage_demo
from src.models import evaluate, train


def main():
    print("\n=== 1/6 Harmonizing real NCRB + Census 2011 data ===")
    harmonize.main()

    print("\n=== 2/6 Building district spatial graph ===")
    spatial_graph.main()

    print("\n=== 3/6 Generating synthetic district-month panel ===")
    synthesize.main()

    print("\n=== 4/6 Training ST-SAGE ===")
    train.main()

    print("\n=== 5/6 Evaluating ST-SAGE on held-out months ===")
    evaluate.evaluate()

    print("\n=== 6/6 Running crime-linkage demo ===")
    run_linkage_demo.main()

    print("\nDone. Launch the dashboard with:")
    print("  .venv/Scripts/streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
