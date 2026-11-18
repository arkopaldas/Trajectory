from verifier import run_verification

if __name__ == "__main__":
    result = run_verification("unicycle")
    print(f"\n[Unicycle Model] Final verdict: {result['verdict']}")
