Write-Host "Running full ST-SAGE pipeline (data -> graph -> synthetic panel -> train -> evaluate -> linkage demo)..."
& ".\.venv\Scripts\python.exe" "scripts\run_pipeline.py"
Write-Host "`nLaunching dashboard at http://localhost:8501 ..."
& ".\.venv\Scripts\streamlit.exe" run "app\dashboard.py"
