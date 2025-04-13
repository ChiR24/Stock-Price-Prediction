@echo off
echo Starting Stock Market Prediction System...

:: Activate the virtual environment
call .venv\Scripts\activate.bat

:: Install required packages if not already installed
pip install pandas python-dotenv matplotlib scikit-learn tensorflow ta plotly dash

:: Run the system with the virtual environment's Python
.venv\Scripts\python.exe run_system.py --all --tickers AAPL MSFT GOOG --data-source yahoo

:: Deactivate the virtual environment
call .venv\Scripts\deactivate.bat

pause
