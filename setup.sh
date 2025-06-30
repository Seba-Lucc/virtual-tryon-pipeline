#!/bin/bash
# Virtual Try-On Pipeline Setup Script

echo "🚀 Setting up Virtual Try-On Pipeline..."

# Navigate to project directory
cd "$(dirname "$0")/modvt"

# Check if virtual environment exists
if [ ! -d "modvt" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv modvt
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source modvt/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install torch torchvision torchaudio
pip install transformers diffusers accelerate
pip install opencv-python pillow numpy
pip install huggingface_hub

echo "✅ Setup complete!"
echo ""
echo "To run the pipeline:"
echo "1. cd modvt"
echo "2. source modvt/bin/activate"
echo "3. python fixed-pipeline.py"
echo ""
echo "To test quickly:"
echo "python quick-test.py"