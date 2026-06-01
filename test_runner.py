#!/usr/bin/env python3
"""
Test runner script for the transformer project.

This script provides an easy way to run the transformer tests with enhanced output.
"""

import subprocess
import sys
import time
import traceback
from pathlib import Path


def print_banner(title, char="=", width=80):
    """Print a formatted banner."""
    print("\n" + char * width)
    print(f"{title:^{width}}")
    print(char * width)


def print_section(title, char="-", width=60):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"🔧 {title}")
    print(f"{char * width}")


def run_unittest():
    """Run tests using the built-in unittest framework."""
    print_section("Running Tests with unittest")
    start_time = time.time()

    result = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_transformer", "-v"],
        cwd=Path(__file__).parent,
        check=False,
    )

    duration = time.time() - start_time

    if result.returncode == 0:
        print(f"\n✅ All tests passed in {duration:.2f}s!")
    else:
        print(f"\n❌ Tests failed after {duration:.2f}s")

    return result.returncode


def run_pytest():
    """Run tests using pytest with beautiful output."""
    print_section("Running Tests with pytest + Rich Output")
    start_time = time.time()

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "-v",
                "--tb=short",
                "--color=yes",
                "--durations=10",
                "--html=reports/report.html",
                "--self-contained-html",
            ],
            cwd=Path(__file__).parent,
            check=False,
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✅ All tests passed in {duration:.2f}s!")
            print("📊 HTML report generated: reports/report.html")
        else:
            print(f"\n❌ Tests failed after {duration:.2f}s")

        return result.returncode

    except FileNotFoundError:
        print("❌ pytest not found. Installing...")
        # Try to install pytest
        install_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "pytest",
                "pytest-html",
                "pytest-cov",
                "pytest-sugar",
                "rich",
            ],
            check=False,
        )
        if install_result.returncode == 0:
            print("✅ pytest installed successfully! Running tests...")
            return run_pytest()
        print("❌ Failed to install pytest. Falling back to unittest...")
        return run_unittest()


def run_pytest_with_coverage():
    """Run tests using pytest with coverage report and beautiful output."""
    print_section("Running Tests with Coverage Analysis + Rich Output")
    start_time = time.time()

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "--cov=.",
                "--cov-report=term-missing",
                "--cov-report=html:reports/coverage",
                "-v",
                "--tb=short",
                "--color=yes",
                "--durations=10",
                "--html=reports/report.html",
                "--self-contained-html",
            ],
            cwd=Path(__file__).parent,
            check=False,
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✅ All tests passed with coverage in {duration:.2f}s!")
            print("📊 HTML report: reports/report.html")
            print("📈 Coverage report: reports/coverage/index.html")
        else:
            print(f"\n❌ Tests failed after {duration:.2f}s")

        return result.returncode

    except FileNotFoundError:
        print("❌ pytest-cov not found. Installing...")
        install_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "pytest",
                "pytest-cov",
                "pytest-html",
                "pytest-sugar",
                "rich",
            ],
            check=False,
        )
        if install_result.returncode == 0:
            print("✅ pytest installed successfully! Running tests...")
            return run_pytest_with_coverage()
        print("❌ Failed to install pytest. Falling back to unittest...")
        return run_unittest()


def run_pytest_parallel():
    """Run tests in parallel using pytest-xdist."""
    print_section("Running Tests in Parallel")
    start_time = time.time()

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "-v",
                "-n",
                "auto",  # Auto-detect number of CPUs
                "--html=reports/report.html",
                "--self-contained-html",
            ],
            cwd=Path(__file__).parent,
            check=False,
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✅ All tests passed in parallel in {duration:.2f}s!")
        else:
            print(f"\n❌ Parallel tests failed after {duration:.2f}s")

        return result.returncode

    except FileNotFoundError:
        print("❌ pytest-xdist not found. Install with: pip install pytest-xdist")
        return run_pytest()


def run_benchmark_tests():
    """Run benchmark tests only."""
    print_section("Running Benchmark Tests")
    start_time = time.time()

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "--benchmark-only",
                "--benchmark-sort=mean",
                "-v",
                "--tb=short",
                "--color=yes",
            ],
            cwd=Path(__file__).parent,
            check=False,
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✅ Benchmark tests completed in {duration:.2f}s!")
        else:
            print(f"\n❌ Benchmark tests failed after {duration:.2f}s")

        return result.returncode

    except FileNotFoundError:
        print(
            "❌ pytest-benchmark not found. Install with: pip install pytest-benchmark"
        )
        return run_pytest()


def run_standalone_test():
    """Run a simple standalone test to check basic functionality."""
    print_section("Running Standalone Test")
    start_time = time.time()

    try:
        # Add the current directory to Python path
        sys.path.insert(0, str(Path(__file__).parent))
        from tests.test_transformer import (
            standalone_test_run,
        )  # pylint: disable=import-outside-toplevel

        config_file = Path("config.toml")
        if not config_file.exists():
            print(f"❌ Configuration file {config_file} not found!")
            return 1

        print("🚀 Starting standalone test...")
        standalone_test_run(config_file)

        duration = time.time() - start_time
        print(f"\n✅ Standalone test completed successfully in {duration:.2f}s!")
        return 0

    except ImportError as e:
        duration = time.time() - start_time
        print(f"\n❌ Import error after {duration:.2f}s: {e}")
        print("💡 Make sure the tests directory contains __init__.py")
        return 1
    except Exception as e:  # pylint: disable=broad-except
        duration = time.time() - start_time
        print(f"\n❌ Error running standalone test after {duration:.2f}s: {e}")
        traceback.print_exc()
        return 1


def show_help():
    """Show help message."""
    print_banner("🔥 TRANSFORMER TEST RUNNER 🔥")
    print(
        """
Available test options:

📋 unittest       - Run tests with Python's built-in unittest
🚀 pytest        - Run tests with pytest + rich output (recommended)
📊 coverage      - Run tests with pytest and generate coverage report
🏃 parallel      - Run tests in parallel using pytest-xdist
⚡ benchmark     - Run benchmark tests only
🎯 standalone    - Run the original standalone test function
❓ help          - Show this help message

Examples:
    python test_runner.py                 # Default: pytest
    python test_runner.py pytest
    python test_runner.py coverage
    python test_runner.py parallel
    python test_runner.py benchmark
    python test_runner.py standalone
    """
    )


def main():
    """Main function to run tests."""
    print_banner("🔥 TRANSFORMER TEST RUNNER 🔥")
    print(f"📅 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()

        if test_type == "help":
            show_help()
            return 0
        if test_type == "pytest":
            return run_pytest()
        if test_type == "unittest":
            return run_unittest()
        if test_type == "coverage":
            return run_pytest_with_coverage()
        if test_type == "parallel":
            return run_pytest_parallel()
        if test_type == "benchmark":
            return run_benchmark_tests()
        if test_type == "standalone":
            return run_standalone_test()
        print(f"❌ Unknown test type: {test_type}")
        print(
            "💡 Available options: unittest, pytest, coverage, parallel, benchmark, standalone, help"
        )
        return 1
    # Default to pytest if no argument provided
    return run_pytest()


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
