"""Entry point: counterexample-guided fine-tuning (CEGIS) of the SIMPLE network until it verifies as safe. See cegis.py.
Run:  python cegis_simple.py      (from inside the Verification folder; writes Networks/<N> Trained/Simple_CEGIS.onnx)
"""
from cegis import run_cegis

if __name__ == "__main__":
    print(f"\n[Simple Model] CEGIS result: {run_cegis('simple')}")
