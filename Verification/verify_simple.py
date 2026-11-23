"""Entry point: verify the trained SIMPLE network. Settings in config.py; prints SAFE / UNSAFE / UNKNOWN, writes the per-cell report
to Verification/Results and appends counterexamples (if any) to Datasets/Counterexamples.
Run:  python verify_simple.py     (from inside the Verification folder; needs maraboupy, onnx, onnxruntime)
"""
from verifier import run_verification

if __name__ == "__main__":
    result = run_verification("simple")
    print(f"\n[Simple Model] Final verdict: {result['verdict']}")
