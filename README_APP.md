# 🎵 Music Genre Classifier - Streamlit App

A beautiful and interactive web application for classifying music genres using deep learning.

## ✨ Features

- **🎯 10 Genre Classification**: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
- **📊 Interactive Visualizations**: Waveforms, confidence scores, and detailed analysis
- **🎧 Audio Playback**: Listen to your uploaded files directly in the app
- **📈 Chunk-by-Chunk Analysis**: See how the AI analyzes different parts of your song
- **🎨 Beautiful UI**: Modern, responsive design with gradient themes

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Trained model file (`Trained_model.keras`)
- Virtual environment with required packages

### Installation & Launch

1. **Easy Launch** (Recommended):
   ```bash
   ./run_app.sh
   ```

2. **Manual Launch**:
   ```bash
   # Activate your virtual environment
   source .venv/bin/activate
   
   # Install requirements
   pip install -r streamlit_requirements.txt
   
   # Run the app
   streamlit run app.py
   ```

3. **Open your browser** and go to `http://localhost:8501`

## 📁 File Structure

```
MusicGenreClassification/
├── app.py                      # Main Streamlit application
├── Trained_model.keras         # Your trained model (required)
├── streamlit_requirements.txt  # App-specific requirements
├── run_app.sh                 # Easy launcher script
└── README_APP.md              # This file
```

## 🎵 How to Use

1. **Upload Audio**: Click "Choose an audio file" in the sidebar
2. **Supported Formats**: WAV, MP3, FLAC, OGG
3. **View Results**: Get instant genre prediction with confidence scores
4. **Explore Analysis**: Check detailed visualizations and chunk analysis

## 🎨 App Sections

### 🏠 Main Dashboard
- **Header**: Beautiful gradient design with app title
- **Upload Area**: Drag & drop or browse for audio files
- **Results Display**: Large, prominent genre prediction

### 📊 Analysis Panel
- **Audio Player**: Listen to your uploaded file
- **Waveform**: Visual representation of the audio signal
- **Confidence Scores**: Bar chart showing all genre probabilities
- **Chunk Distribution**: Pie chart of chunk-level predictions

### 🔍 Detailed Analysis
- **Metrics**: Quick stats (confidence, duration, sample rate)
- **Top Predictions**: Top 3 most likely genres
- **Chunk Table**: Detailed breakdown of each audio segment

## 🎛️ Sidebar Features
- **File Upload**: Main upload interface
- **Genre Guide**: Visual list of all supported genres
- **How It Works**: Step-by-step process explanation

## 🔧 Technical Details

### Audio Processing
- **Chunk Size**: 4-second segments
- **Overlap**: 2-second overlap between chunks
- **Features**: Mel spectrograms (150x150 pixels)
- **Aggregation**: Majority voting across chunks

### Model Architecture
- **Input**: Mel spectrogram images
- **Output**: 10 genre probabilities
- **Format**: TensorFlow/Keras model

## 🎨 Design Features
- **Responsive Layout**: Works on desktop and mobile
- **Custom CSS**: Beautiful gradients and animations
- **Interactive Charts**: Plotly visualizations
- **Genre Colors**: Unique color scheme for each genre
- **Emoji Icons**: Visual genre identifiers

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**:
   - Ensure `Trained_model.keras` is in the same directory as `app.py`

2. **Import errors**:
   - Run: `pip install -r streamlit_requirements.txt`

3. **Audio format not supported**:
   - Convert to WAV, MP3, FLAC, or OGG format

4. **App won't start**:
   - Check if port 8501 is available
   - Try: `streamlit run app.py --server.port 8502`

### Performance Tips
- **File Size**: Smaller files (< 10MB) process faster
- **Duration**: 30 seconds to 5 minutes work best
- **Quality**: Higher quality audio may improve accuracy

## 📱 Mobile Support
The app is responsive and works on mobile devices, though desktop is recommended for the best experience.

## 🎵 Example Files
Place test audio files in a `TestMusic/` folder for easy access during development.

---

**🎶 Enjoy classifying your music! 🎶**
