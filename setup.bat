@echo off
REM Virtual Try-On Pipeline Setup Script for Windows

echo 🚀 Setting up Virtual Try-On Pipeline...

REM Navigate to project directory
cd /d "%~dp0\modvt"

REM Check if virtual environment exists
if not exist "modvt" (
    echo 📦 Creating virtual environment...
    python -m venv modvt
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call modvt\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📚 Installing dependencies...
pip install torch torchvision torchaudio
pip install transformers diffusers accelerate
pip install opencv-python pillow numpy
pip install huggingface_hub

echo ✅ Setup complete!
echo.
echo To run the pipeline:
echo 1. cd modvt
echo 2. modvt\Scripts\activate.bat
echo 3. python fixed-pipeline.py
echo.
echo To test quickly:
echo python quick-test.py

pause