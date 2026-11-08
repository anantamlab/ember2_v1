@echo off
rem P1-09 single confirmation launcher. Runs the frozen command exactly once; no retry.
set "OUT=C:\Users\sanka\AppData\Local\Temp\claude\1817b1d6ffc1cf67\45b40714-8984-4bd7-b28d-8bebfcaf89ff\scratchpad\run"
set "PATH=C:\Program Files\Git\cmd;%PATH%"
set "PYTHONUNBUFFERED=1"
cd /d D:\Coding\ember_v1-p10
if exist "data\selection\p1-09-selection@1\attempt.json" (
  echo REFUSED: attempt marker exists > "%OUT%\run_exit.txt"
  exit /b 9
)
echo %DATE% %TIME% > "%OUT%\run_started.txt"
D:\Coding\ember_v1-p10\.venv\Scripts\python.exe -m ember_v1.teacher.confirmation run --protocol-commit 64b2d9873953ba750fd6c08fb84761f1bcd8e0bb > "%OUT%\run_stdout.txt" 2> "%OUT%\run_stderr.txt"
echo EXIT %ERRORLEVEL% %DATE% %TIME% > "%OUT%\run_exit.txt"
