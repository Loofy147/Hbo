import argparse
import sys
from importlib import import_module

def main():
    """
    Command-line interface for running HPO engine examples.
    """
    parser = argparse.ArgumentParser(
        description="🚀 HPO Engine: Command-Line Interface",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "example",
        nargs='?',
        default="all",
        help=(
            "The example to run. Choose from:\n"
            "  - 'math': Run the basic mathematical optimization example.\n"
            "  - 'sklearn': Run the Scikit-learn RandomForest optimization example.\n"
            "  - 'all' (default): Run all available examples sequentially."
        )
    )

    args = parser.parse_args()

    # --- Example runners ---
    def run_math_example():
        print("\n" + "="*50)
        print("    🚀 Running: Mathematical Optimization Example")
        print("="*50)
        try:
            math_module = import_module("examples.01_mathematical_optimization")
            math_module.main()
            print("✅ Mathematical example finished successfully.")
        except ImportError:
            print("❌ Error: Could not import the example. Make sure 'hpo' is installed correctly.")
        except Exception as e:
            print(f"❌ An error occurred during the mathematical example: {e}")

    def run_sklearn_example():
        print("\n" + "="*50)
        print("    🚀 Running: Scikit-learn Optimization Example")
        print("="*50)
        try:
            sklearn_module = import_module("examples.02_sklearn_optimization")
            sklearn_module.main()
            print("✅ Scikit-learn example finished successfully.")
        except ImportError as e:
            if "sklearn" in str(e):
                print("❌ Error: Scikit-learn is not installed. Please run 'pip install scikit-learn'.")
            else:
                print(f"❌ Error: Could not import the example: {e}. Make sure 'hpo' is installed correctly.")
        except Exception as e:
            print(f"❌ An error occurred during the sklearn example: {e}")


    # --- Execute selected example ---
    if args.example == 'math':
        run_math_example()
    elif args.example == 'sklearn':
        run_sklearn_example()
    elif args.example == 'all':
        run_math_example()
        run_sklearn_example()
        print("\n" + "="*50)
        print("🎉 All examples completed.")
        print("="*50)
    else:
        print(f"Unknown example: '{args.example}'")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    # To make the examples runnable from the root directory before installation,
    # we add the 'src' directory to the Python path.
    import os
    sys.path.insert(0, os.path.abspath('./src'))
    main()