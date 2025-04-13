#!/usr/bin/env python
"""
Script to run the entire stock market prediction system.
This script starts all components of the system.
"""
import os
import sys

# Add the virtual environment's site-packages to the Python path
venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'Lib', 'site-packages')
if os.path.exists(venv_path):
    sys.path.insert(0, venv_path)

import argparse
import subprocess
import time
import signal
import threading
from datetime import datetime

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Stock Market Prediction System')

    # System components
    parser.add_argument('--data-collection', action='store_true',
                        help='Run data collection component')
    parser.add_argument('--real-time-prediction', action='store_true',
                        help='Run real-time prediction component')
    parser.add_argument('--dashboard', action='store_true',
                        help='Run visualization dashboard')
    parser.add_argument('--all', action='store_true',
                        help='Run all components')

    # Data collection parameters
    parser.add_argument('--tickers', nargs='+', default=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                        help='List of stock tickers to analyze')
    parser.add_argument('--data-source', choices=['yahoo', 'alpha_vantage'], default='yahoo',
                        help='Source for stock data (yahoo or alpha_vantage)')

    # Prediction parameters
    parser.add_argument('--model-type', choices=['lstm', 'ensemble', 'all'], default='all',
                        help='Type of model to use for prediction')

    return parser.parse_args()

def run_data_collection(args):
    """Run the data collection component."""
    print("Starting data collection component...")

    # Use the virtual environment's Python executable
    python_executable = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'Scripts', 'python.exe')
    if not os.path.exists(python_executable):
        python_executable = 'python'  # Fallback to system Python

    # Build command
    cmd = [
        python_executable, "src/main.py",
        "--tickers", *args.tickers,
        "--data-source", args.data_source,
        "--use-spark"
    ]

    # Run command
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    return process

def run_real_time_prediction(args):
    """Run the real-time prediction component."""
    print("Starting real-time prediction component...")

    # Use the virtual environment's Python executable
    python_executable = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'Scripts', 'python.exe')
    if not os.path.exists(python_executable):
        python_executable = 'python'  # Fallback to system Python

    # Build command
    cmd = [
        python_executable, "src/run_real_time_prediction.py",
        "--tickers", *args.tickers,
        "--data-source", args.data_source,
        "--model-type", args.model_type,
        "--use-kafka",
        "--use-mongodb"
    ]

    # Run command
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    return process

def run_dashboard():
    """Run the visualization dashboard."""
    print("Starting visualization dashboard...")

    # Use the virtual environment's Python executable
    python_executable = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'Scripts', 'python.exe')
    if not os.path.exists(python_executable):
        python_executable = 'python'  # Fallback to system Python

    # Get the virtual environment's pip executable
    pip_executable = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'Scripts', 'pip.exe')
    if not os.path.exists(pip_executable):
        pip_executable = 'pip'  # Fallback to system pip

    # Install the correct version of typing_extensions
    print("Installing the correct version of typing_extensions...")
    subprocess.run([pip_executable, "install", "typing_extensions==4.7.1"], check=True)

    # Install dash with the correct dependencies
    print("Installing dash with the correct dependencies...")
    subprocess.run([pip_executable, "install", "dash==2.9.3"], check=True)

    # Build command
    cmd = [
        python_executable, "src/visualization/app.py"
    ]

    # Run command
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    return process

def log_output(process, name):
    """Log the output of a process."""
    for line in process.stdout:
        print(f"[{name}] {line.strip()}")

def main():
    """Main function to run the system."""
    # Parse command line arguments
    args = parse_arguments()

    # Set default if no component specified
    if not (args.data_collection or args.real_time_prediction or args.dashboard or args.all):
        args.all = True

    # Start processes
    processes = []

    try:
        # Start data collection
        if args.data_collection or args.all:
            data_collection_process = run_data_collection(args)
            processes.append(("Data Collection", data_collection_process))

            # Start output logging thread
            threading.Thread(
                target=log_output,
                args=(data_collection_process, "Data Collection"),
                daemon=True
            ).start()

            # Wait a bit for data collection to start
            time.sleep(5)

        # Start real-time prediction
        if args.real_time_prediction or args.all:
            real_time_prediction_process = run_real_time_prediction(args)
            processes.append(("Real-Time Prediction", real_time_prediction_process))

            # Start output logging thread
            threading.Thread(
                target=log_output,
                args=(real_time_prediction_process, "Real-Time Prediction"),
                daemon=True
            ).start()

            # Wait a bit for real-time prediction to start
            time.sleep(5)

        # Start dashboard
        if args.dashboard or args.all:
            dashboard_process = run_dashboard()
            processes.append(("Dashboard", dashboard_process))

            # Start output logging thread
            threading.Thread(
                target=log_output,
                args=(dashboard_process, "Dashboard"),
                daemon=True
            ).start()

        print("All components started. Press Ctrl+C to stop.")

        # Wait for processes to finish
        while True:
            time.sleep(1)

            # Check if any process has terminated
            for name, process in processes:
                if process.poll() is not None:
                    print(f"{name} process terminated with exit code {process.returncode}")

                    # If any process terminates, stop all processes
                    if process.returncode != 0:
                        print("Stopping all processes due to component failure...")
                        for _, p in processes:
                            if p.poll() is None:
                                p.terminate()
                        return 1

    except KeyboardInterrupt:
        print("Stopping all processes...")

        # Stop all processes
        for name, process in processes:
            if process.poll() is None:
                print(f"Stopping {name} process...")
                process.terminate()

        # Wait for processes to terminate
        for name, process in processes:
            try:
                process.wait(timeout=5)
                print(f"{name} process terminated")
            except subprocess.TimeoutExpired:
                print(f"{name} process did not terminate gracefully, killing...")
                process.kill()

    return 0

if __name__ == "__main__":
    sys.exit(main())
