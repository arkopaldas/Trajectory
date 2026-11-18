from verifier import run_verification

if __name__ == "__main__":
    result = run_verification("simple")
    print(f"\n[Simple Model] Final verdict: {result['verdict']}")
