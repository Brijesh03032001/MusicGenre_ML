# 🎵 Music Genre Classifier

An AI-powered web application that classifies music genres using deep learning.

## Features

- 🎯 **Precise Classification**: Identifies 10 different music genres with high accuracy
- 📊 **Detailed Analysis**: Provides confidence scores and chunk-by-chunk breakdown
- 🎵 **Multiple Formats**: Supports WAV, MP3, FLAC, and OGG audio files
- 📱 **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- 🔒 **Privacy First**: Audio files are processed locally and temporarily

## Supported Genres

- Blues 🎸
- Classical 🎼
- Country 🤠
- Disco 🕺
- Hip-hop 🎤
- Jazz 🎺
- Metal 🤘
- Pop 🎵
- Reggae 🏝️
- Rock 🎸

## How to Run Locally

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deployment

This app is ready for deployment on:
- Streamlit Community Cloud
- Heroku
- Docker containers
- Any cloud platform supporting Python web apps

## Technology Stack

- **Frontend**: Streamlit
- **ML Framework**: TensorFlow/Keras
- **Audio Processing**: Librosa
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy

## Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: Mel spectrograms (148x149)
- **Processing**: 4-second audio chunks with 2-second overlap
- **Output**: 10 genre classifications

Built with ❤️ using Streamlit and TensorFlow
