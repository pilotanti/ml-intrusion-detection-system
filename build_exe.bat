@echo off
REM Build IDS.exe - a single-file Windows executable of the IDS desktop app.
REM Usage: build_exe.bat   (run from the project root)
setlocal
cd /d "%~dp0"

python -m pip install -r requirements.txt || goto :error

if not exist "data\KDDTrain+.txt" python src\download_data.py || goto :error
if not exist "models\ids_model.joblib" python src\train_model.py || goto :error

python -m PyInstaller --noconfirm --clean --onefile --windowed ^
  --name IDS ^
  --icon assets\ids.ico ^
  --paths src ^
  --add-data "models\ids_model.joblib;models" ^
  --add-data "samples\sample_traffic.csv;samples" ^
  --add-data "assets\ids.ico;assets" ^
  --collect-submodules sklearn.ensemble --collect-submodules sklearn.tree ^
  --collect-submodules scipy._external.array_api_compat ^
  --collect-submodules scipy._external.array_api_extra ^
  --collect-submodules sklearn.externals.array_api_compat ^
  --collect-submodules sklearn.externals.array_api_extra ^
  --exclude-module matplotlib --exclude-module IPython --exclude-module pytest ^
  --exclude-module PyQt5 --exclude-module PySide6 --exclude-module notebook ^
  src\ids_app.py || goto :error

echo.
echo Build complete: dist\IDS.exe
exit /b 0

:error
echo Build failed.
exit /b 1
