@echo off
setlocal

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller

pyinstaller --noconfirm --onefile --windowed --name fpcb-heatmap-app main.py

echo Build complete. Check dist\fpcb-heatmap-app.exe

