@echo off
REM Script to activate the Python virtual environment

REM --- Configuration ---
REM Set the path to your repository (where this script is located)
SET "REPO_PATH=%~dp0"

REM Set the name of your virtual environment folder
SET "VENV_FOLDER=venv"  REM Common names: .venv, venv, myenv, etc.

REM --- End Configuration ---

REM Construct the full path to the activation script
SET "VENV_ACTIVATE_SCRIPT=%REPO_PATH%%VENV_FOLDER%\Scripts\activate.bat"

REM Check if the activation script exists
IF NOT EXIST "%VENV_ACTIVATE_SCRIPT%" (
    echo ERROR: Virtual environment activation script not found!
    echo Expected at: %VENV_ACTIVATE_SCRIPT%
    echo Please check your VENV_FOLDER configuration in this script.
    pause
    exit /b 1
)

REM Activate the virtual environment
echo Activating virtual environment...
CALL "%VENV_ACTIVATE_SCRIPT%"

REM You are now in the virtual environment.
REM The script will keep the command prompt open.
echo.
echo Virtual environment '%VENV_FOLDER%' is now active.
echo Type 'deactivate' to exit the virtual environment.
echo Type 'exit' to close this window when done.
echo.

REM Optionally, navigate to the repository path if not already there
cd /D "%REPO_PATH%"

REM Open a new command prompt window with the activated environment
REM This allows the current script to finish, but leaves you in the venv.
cmd /k