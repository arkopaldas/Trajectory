"""Entry point: counterexample-guided fine-tuning (CEGIS) of the UNICYCLE network until it verifies as safe. See cegis.py.
Run:  python cegis_unicycle.py    (from inside the Verification folder; writes Networks/<N> Trained/Unicycle_CEGIS.onnx)
"""
from cegis import run_cegis

if __name__ == "__main__":
    print(f"\n[Unicycle Model] CEGIS result: {run_cegis('unicycle')}")
