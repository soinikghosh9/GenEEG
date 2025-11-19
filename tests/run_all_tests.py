"""
Comprehensive Test Runner for GenEEG

This script runs all unit tests and provides a summary of results.

Usage:
    python tests/run_all_tests.py
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_test_file(test_file, project_root):
    """Run a single test file and return results."""
    print(f"\n{'='*80}")
    print(f"Running: {test_file.name}")
    print(f"{'='*80}")
    
    try:
        # Set up environment with PYTHONPATH pointing to project root
        env = os.environ.copy()
        current_pythonpath = env.get('PYTHONPATH', '')
        if current_pythonpath:
            env['PYTHONPATH'] = f"{project_root}{os.pathsep}{current_pythonpath}"
        else:
            env['PYTHONPATH'] = str(project_root)
        
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=env,
            cwd=str(project_root)  # Run from project root
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {test_file.name} took too long")
        return False
    except Exception as e:
        print(f"[ERROR] running {test_file.name}: {e}")
        return False

def main():
    """Run all test files."""
    tests_dir = Path(__file__).parent
    project_root = tests_dir.parent
    
    # Find all test files
    test_files = sorted(tests_dir.glob("test_*.py"))
    
    if not test_files:
        print("[ERROR] No test files found!")
        return 1
    
    print(f"\n{'#'*80}")
    print(f"# GenEEG Test Suite - Found {len(test_files)} test files")
    print(f"{'#'*80}\n")
    
    results = {}
    
    for test_file in test_files:
        success = run_test_file(test_file, project_root)
        results[test_file.name] = success
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}\n")
    
    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed
    
    for test_name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status}: {test_name}")
    
    print(f"\n{'='*80}")
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print(f"Success Rate: {100*passed/len(results):.1f}%")
    print(f"{'='*80}\n")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
