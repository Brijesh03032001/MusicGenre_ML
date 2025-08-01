#!/bin/bash

# Music Genre Classifier - Streamlit App Launcher
echo "🎵 Starting Music Genre Classifier..."
echo "📁 Current directory: $(pwd)"

# Check if trained model exists
if [ ! -f "Trained_model.keras" ]; then
    echo "❌ Error: Trained_model.keras not found!"
    echo "Please ensure your trained model file is in the current directory."
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not detected. Activating..."
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        echo "✅ Virtual environment activated"
    else
        echo "❌ No virtual environment found. Please activate your environment first."
        exit 1
    fi
fi

# Install additional requirements if needed
echo "📦 Installing Streamlit app requirements..."
pip install -r streamlit_requirements.txt

# Launch the app
echo "🚀 Launching Streamlit app..."
echo "🌐 The app will open in your default browser"
echo "🔗 URL: http://localhost:8501"
echo ""
echo "📝 To stop the app, press Ctrl+C in this terminal"
echo ""

streamlit run app.py
