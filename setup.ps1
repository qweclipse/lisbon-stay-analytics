$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
py -3.12 -m venv .venv
if ($LASTEXITCODE) { throw 'Python environment failed' }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE) { throw 'Dependency installation failed' }
