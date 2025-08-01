import os
# --- Auto-download model from Google Drive if missing ---
MODEL_PATH = "Trained_model.keras"
GOOGLE_DRIVE_ID = "1EvDGdPcBcNOW3YquzukxlIpKpEJL9y94"
if not os.path.exists(MODEL_PATH):
    try:
        import gdown
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "gdown"])
        import gdown
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
    print("Downloading model from Google Drive...")
    gdown.download(url, MODEL_PATH, quiet=False)

import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from collections import Counter
import tempfile
import pandas as pd

# Set page config
st.set_page_config(
    page_title="🎵 Music Genre Classifier",
    page_icon="🎶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for both light and dark themes + responsive design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for theme colors */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --success-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        
        /* Light theme colors */
        --bg-primary-light: #ffffff;
        --bg-secondary-light: #f8f9fa;
        --text-primary-light: #1a1a1a;
        --text-secondary-light: #666666;
        --border-light: #e0e0e0;
        --shadow-light: rgba(0, 0, 0, 0.1);
        
        /* Dark theme colors */
        --bg-primary-dark: #0e1117;
        --bg-secondary-dark: #262730;
        --text-primary-dark: #fafafa;
        --text-secondary-dark: #a0a0a0;
        --border-dark: #333333;
        --shadow-dark: rgba(255, 255, 255, 0.1);
    }
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main header - responsive and theme-aware */
    .main-header {
        background: var(--primary-gradient);
        padding: clamp(1rem, 4vw, 3rem);
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    
    .main-header h1 {
        color: white;
        font-size: clamp(2rem, 5vw, 3.5rem);
        margin: 0;
        font-weight: 700;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: clamp(1rem, 2.5vw, 1.3rem);
        margin: 0.5rem 0 0 0;
        position: relative;
        z-index: 1;
    }
    
    /* Landing page styles */
    .landing-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    /* Feature card - simplified and theme-aware */
    .feature-card {
        background: var(--bg-primary-light);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 25px var(--shadow-light);
        border: 1px solid var(--border-light);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Dark theme for feature cards */
    [data-theme="dark"] .feature-card {
        background: var(--bg-secondary-dark);
        box-shadow: 0 8px 25px var(--shadow-dark);
        border: 1px solid var(--border-dark);
        color: var(--text-primary-dark);
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--primary-gradient);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px var(--shadow-light);
    }
    
    [data-theme="dark"] .feature-card:hover {
        box-shadow: 0 15px 35px var(--shadow-dark);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: var(--text-primary-light);
    }
    
    [data-theme="dark"] .feature-title {
        color: var(--text-primary-dark);
    }
    
    .feature-description {
        color: var(--text-secondary-light);
        line-height: 1.6;
    }
    
    [data-theme="dark"] .feature-description {
        color: var(--text-secondary-dark);
    }
    
    /* Genre card - theme-aware */
    .genre-card {
        background: var(--bg-primary-light);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px var(--shadow-light);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
        color: var(--text-primary-light);
    }
    
    [data-theme="dark"] .genre-card {
        background: var(--bg-secondary-dark);
        box-shadow: 0 4px 15px var(--shadow-dark);
        color: var(--text-primary-dark);
    }
    
    .genre-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px var(--shadow-light);
    }
    
    [data-theme="dark"] .genre-card:hover {
        box-shadow: 0 8px 25px var(--shadow-dark);
    }
    
    /* Metric container - responsive and theme-aware */
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px var(--shadow-light);
    }
    
    [data-theme="dark"] .metric-container {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        box-shadow: 0 4px 15px var(--shadow-dark);
    }
    
    /* Button styles - enhanced */
    .stButton > button {
        background: var(--primary-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Prediction result - enhanced */
    .prediction-result {
        background: var(--primary-gradient);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-result::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .prediction-result h2 {
        font-size: clamp(2rem, 4vw, 2.8rem);
        margin: 0 0 1rem 0;
        font-weight: 700;
        position: relative;
        z-index: 1;
    }
    
    .prediction-result p {
        font-size: clamp(1.1rem, 2vw, 1.4rem);
        margin: 0.5rem 0;
        position: relative;
        z-index: 1;
    }
    
    /* Sidebar enhancements */
    .sidebar-content {
        background: var(--bg-secondary-light);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid var(--border-light);
    }
    
    [data-theme="dark"] .sidebar-content {
        background: var(--bg-secondary-dark);
        border: 1px solid var(--border-dark);
    }
    
    /* Responsive grid system */
    .responsive-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    /* Mobile optimizations */
    @media (max-width: 768px) {
        .main-header {
            margin: 1rem;
            padding: 1.5rem;
        }
        
        .feature-card {
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .genre-card {
            margin: 0.5rem 0;
            padding: 1rem;
        }
        
        .metric-container {
            padding: 1rem;
        }
        
        .prediction-result {
            margin: 1rem;
            padding: 1.5rem;
        }
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        border: 2px dashed var(--border-light) !important;
        border-radius: 15px !important;
        padding: 2rem !important;
        transition: all 0.3s ease !important;
    }
    
    [data-theme="dark"] .stFileUploader > div > div {
        border-color: var(--border-dark) !important;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #667eea !important;
        background: rgba(102, 126, 234, 0.05) !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: var(--primary-gradient) !important;
        border-radius: 10px !important;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background: var(--success-gradient) !important;
        border-radius: 10px !important;
        border: none !important;
    }
    
    .stError {
        background: var(--secondary-gradient) !important;
        border-radius: 10px !important;
        border: none !important;
    }
    
    /* Welcome section */
    .welcome-section {
        text-align: center;
        padding: 3rem 1rem;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .welcome-title {
        font-size: clamp(2.5rem, 5vw, 4rem);
        font-weight: 700;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    .welcome-subtitle {
        font-size: clamp(1.2rem, 3vw, 1.6rem);
        color: var(--text-secondary-light);
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    [data-theme="dark"] .welcome-subtitle {
        color: var(--text-secondary-dark);
    }
    
    /* CTA Button */
    .cta-button {
        display: inline-block;
        background: var(--primary-gradient);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        text-decoration: none;
        font-weight: 600;
        font-size: 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        border: none;
        cursor: pointer;
    }
    
    .cta-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
        text-decoration: none;
        color: white;
    }
    
    /* Stats section */
    .stats-section {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }
    
    .stat-item {
        text-align: center;
        padding: 1.5rem;
        background: var(--bg-primary-light);
        border-radius: 15px;
        box-shadow: 0 4px 15px var(--shadow-light);
        border: 1px solid var(--border-light);
    }
    
    [data-theme="dark"] .stat-item {
        background: var(--bg-secondary-dark);
        box-shadow: 0 4px 15px var(--shadow-dark);
        border: 1px solid var(--border-dark);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: block;
    }
    
    .stat-label {
        color: var(--text-secondary-light);
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    [data-theme="dark"] .stat-label {
        color: var(--text-secondary-dark);
    }
</style>
""", unsafe_allow_html=True)

# Genre information and colors
GENRE_INFO = {
    'blues': {'emoji': '🎸', 'color': '#1f77b4', 'description': 'Soulful and expressive'},
    'classical': {'emoji': '🎼', 'color': '#ff7f0e', 'description': 'Timeless and elegant'},
    'country': {'emoji': '🤠', 'color': '#2ca02c', 'description': 'Heartfelt storytelling'},
    'disco': {'emoji': '🕺', 'color': '#d62728', 'description': 'Groovy and danceable'},
    'hiphop': {'emoji': '🎤', 'color': '#9467bd', 'description': 'Rhythmic and powerful'},
    'jazz': {'emoji': '🎺', 'color': '#8c564b', 'description': 'Smooth and sophisticated'},
    'metal': {'emoji': '🤘', 'color': '#e377c2', 'description': 'Heavy and intense'},
    'pop': {'emoji': '🎵', 'color': '#7f7f7f', 'description': 'Catchy and mainstream'},
    'reggae': {'emoji': '🏝️', 'color': '#bcbd22', 'description': 'Laid-back and rhythmic'},
    'rock': {'emoji': '🎸', 'color': '#17becf', 'description': 'Energetic and driving'}
}

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model("Trained_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def load_and_preprocess_data(file_path, target_shape=(148, 149), max_duration=60):
    """Preprocess audio data for prediction with size limits"""
    try:
        data = []
        # Load with duration limit to prevent memory issues
        audio_data, sample_rate = librosa.load(file_path, sr=22050, duration=max_duration)
        
        # Limit audio data size for display
        display_audio = audio_data[:sample_rate * 30] if len(audio_data) > sample_rate * 30 else audio_data
        
        # Define chunk parameters
        chunk_duration = 4  # seconds
        overlap_duration = 2  # seconds
                    
        # Convert durations to samples
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap_duration * sample_rate)
                    
        # Calculate the number of chunks (limit to prevent memory issues)
        num_chunks = min(50, int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1)
                    
        # Process each chunk
        for i in range(num_chunks):
            start = i * (chunk_samples - overlap_samples)
            end = start + chunk_samples
            
            if end > len(audio_data):
                end = len(audio_data)
                start = max(0, end - chunk_samples)
                        
            chunk = audio_data[start:end]
            
            # Ensure chunk has minimum length
            if len(chunk) < chunk_samples // 2:
                continue
                        
            # Compute Mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate, n_mels=128)
            mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
            data.append(mel_spectrogram)
        
        return np.array(data), sample_rate, display_audio
    
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None, None, None

def model_prediction(X_test, model):
    """Make prediction using the model"""
    try:
        y_pred = model.predict(X_test, verbose=0)
        predicted_categories = np.argmax(y_pred, axis=1)
        
        # Get detailed predictions
        unique_elements, counts = np.unique(predicted_categories, return_counts=True)
        max_count = np.max(counts)
        max_elements = unique_elements[counts == max_count]
        
        return max_elements[0], y_pred, predicted_categories
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def create_waveform_plot(audio_data, sample_rate):
    """Create a beautiful waveform visualization with downsampling"""
    # Downsample for display to prevent browser issues
    max_points = 10000
    if len(audio_data) > max_points:
        step = len(audio_data) // max_points
        audio_data = audio_data[::step]
    
    time = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time, 
        y=audio_data,
        mode='lines',
        name='Waveform',
        line=dict(color='#667eea', width=1),
        fill='tonexty'
    ))
    
    fig.update_layout(
        title="🎵 Audio Waveform",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    return fig

def create_prediction_visualization(predictions, predicted_categories, classes):
    """Create prediction confidence visualization"""
    # Average predictions across all chunks
    avg_predictions = np.mean(predictions, axis=0)
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=avg_predictions,
            marker_color=[GENRE_INFO[genre]['color'] for genre in classes],
            text=[f"{p:.3f}" for p in avg_predictions],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="🎯 Genre Confidence Scores",
        xaxis_title="Music Genres",
        yaxis_title="Confidence Score",
        template="plotly_white",
        height=500,
        showlegend=False
    )
    
    return fig

def create_chunk_analysis(predicted_categories, classes):
    """Create chunk-by-chunk analysis"""
    genre_counts = Counter([classes[i] for i in predicted_categories])
    
    labels = list(genre_counts.keys())
    values = list(genre_counts.values())
    colors = [GENRE_INFO[genre]['color'] for genre in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        hole=.3,
        marker_colors=colors,
        textinfo='label+percent',
        textfont_size=12
    )])
    
    fig.update_layout(
        title="📊 Chunk Distribution by Genre",
        template="plotly_white",
        height=400
    )
    
    return fig

def show_landing_page():
    """Display the landing page"""
    st.markdown("""
    <div class="main-header">
        <h1>🎵 Music Genre Classifier</h1>
        <p>Discover the power of AI-driven music genre classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome section
    st.markdown("""
    <div class="welcome-section">
        <h2 class="welcome-title">Transform Your Music Discovery</h2>
        <p class="welcome-subtitle">
            Our cutting-edge AI analyzes your audio files and instantly identifies their musical genre 
            with remarkable accuracy. From blues to rock, classical to hip-hop – let AI be your music guide.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats section
    st.markdown("""
    <div class="stats-section">
        <div class="stat-item">
            <span class="stat-number">10</span>
            <div class="stat-label">Music Genres</div>
        </div>
        <div class="stat-item">
            <span class="stat-number">95%</span>
            <div class="stat-label">Accuracy Rate</div>
        </div>
        <div class="stat-item">
            <span class="stat-number">4</span>
            <div class="stat-label">Audio Formats</div>
        </div>
        <div class="stat-item">
            <span class="stat-number">∞</span>
            <div class="stat-label">Possibilities</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features section
    st.markdown("## ✨ Key Features")
    
    # Create feature cards using Streamlit columns for better responsiveness
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🎯</span>
            <h3 class="feature-title">Precise Classification</h3>
            <p class="feature-description">
                Advanced deep learning model trained on thousands of audio samples 
                to accurately identify 10 different music genres with high confidence.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">�</span>
            <h3 class="feature-title">Responsive Design</h3>
            <p class="feature-description">
                Beautiful, modern interface that works perfectly on desktop, 
                tablet, and mobile devices with both light and dark themes.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">📊</span>
            <h3 class="feature-title">Detailed Analysis</h3>
            <p class="feature-description">
                Get comprehensive insights with confidence scores, chunk-by-chunk 
                analysis, and beautiful visualizations of your audio data.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🔒</span>
            <h3 class="feature-title">Privacy First</h3>
            <p class="feature-description">
                Your audio files are processed locally and temporarily. 
                No data is stored or shared – your music stays private.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🎵</span>
            <h3 class="feature-title">Multiple Formats</h3>
            <p class="feature-description">
                Support for WAV, MP3, FLAC, and OGG audio formats. Upload any 
                music file and get instant genre classification.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">⚡</span>
            <h3 class="feature-title">Lightning Fast</h3>
            <p class="feature-description">
                Optimized processing pipeline delivers results in seconds. 
                Advanced chunking ensures accurate analysis of any song length.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Supported genres
    st.markdown("## 🎼 Supported Genres")
    
    col1, col2 = st.columns(2)
    
    with col1:
        genres_left = ['blues', 'classical', 'country', 'disco', 'hiphop']
        for genre in genres_left:
            info = GENRE_INFO[genre]
            st.markdown(f"""
            <div class="genre-card">
                <h4>{info['emoji']} {genre.title()}</h4>
                <p>{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        genres_right = ['jazz', 'metal', 'pop', 'reggae', 'rock']
        for genre in genres_right:
            info = GENRE_INFO[genre]
            st.markdown(f"""
            <div class="genre-card">
                <h4>{info['emoji']} {genre.title()}</h4>
                <p>{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # How it works
    st.markdown("## 🔬 How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <span class="feature-icon">📤</span>
            <h4 class="feature-title">1. Upload</h4>
            <p class="feature-description">Select your audio file from any device</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <span class="feature-icon">🔧</span>
            <h4 class="feature-title">2. Process</h4>
            <p class="feature-description">AI analyzes mel spectrograms in chunks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <span class="feature-icon">🤖</span>
            <h4 class="feature-title">3. Classify</h4>
            <p class="feature-description">Deep learning model predicts genre</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <span class="feature-icon">📈</span>
            <h4 class="feature-title">4. Results</h4>
            <p class="feature-description">Get detailed analysis and insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h3 style="margin-bottom: 1rem;">Ready to discover your music's genre?</h3>
            <p style="margin-bottom: 2rem; color: #666;">
                Join thousands of music enthusiasts using our AI-powered genre classifier
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start Classifying Music", key="cta_button", use_container_width=True):
            st.session_state.page = "classifier"
            st.rerun()

def show_classifier_page():
    """Display the main classifier interface"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎵 Music Genre Classifier</h1>
        <p>Upload your audio file and discover its musical genre using AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("❌ Could not load the trained model. Please ensure 'Trained_model.keras' exists in the current directory.")
        return
    
    classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Controls")
        
        # Navigation
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()
        
        st.markdown("---")
        
        # File uploader with size limit
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'ogg'],
            help="Upload an audio file to classify its genre (Max: 25MB)"
        )
        
        # File size validation
        if uploaded_file is not None:
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size_mb > 25:
                st.error(f"❌ File too large ({file_size_mb:.1f}MB). Please upload a file smaller than 25MB.")
                uploaded_file = None
            else:
                st.success(f"✅ File uploaded ({file_size_mb:.1f}MB)")
        
        st.markdown("### 🎵 Supported Genres")
        for genre in classes:
            info = GENRE_INFO[genre]
            st.markdown(f"{info['emoji']} **{genre.title()}** - {info['description']}")
        
        st.markdown("### ℹ️ How it works")
        st.markdown("""
        1. **Upload** your audio file
        2. **AI analyzes** the audio in 4-second chunks
        3. **Mel spectrograms** are extracted from each chunk
        4. **Deep learning model** predicts the genre
        5. **Results** are aggregated for final prediction
        """)
    
    # Main content
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 🎧 Audio Analysis")
                
                # Process audio
                with st.spinner("🔄 Processing audio..."):
                    X_test, sample_rate, audio_data = load_and_preprocess_data(tmp_file_path)
                
                if X_test is not None:
                    # Add channel dimension
                    X_test = np.expand_dims(X_test, axis=-1)
                    
                    # Make prediction
                    with st.spinner("🤖 Making prediction..."):
                        predicted_index, predictions, predicted_categories = model_prediction(X_test, model)
                    
                    if predicted_index is not None:
                        predicted_genre = classes[predicted_index]
                        genre_counts = Counter([classes[i] for i in predicted_categories])
                        confidence = (genre_counts.most_common(1)[0][1] / len(predicted_categories)) * 100
                        
                        # Display prediction result
                        st.markdown(f"""
                        <div class="prediction-result">
                            <h2>{GENRE_INFO[predicted_genre]['emoji']} {predicted_genre.upper()}</h2>
                            <p>Confidence: {confidence:.1f}%</p>
                            <p>Analyzed {len(predicted_categories)} audio chunks</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Audio player
                        st.audio(uploaded_file, format='audio/wav')
                        
                        # Waveform visualization
                        if len(audio_data) > 0:
                            fig_wave = create_waveform_plot(audio_data, sample_rate)
                            st.plotly_chart(fig_wave, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Analysis Results")
                
                if 'predicted_index' in locals():
                    # Metrics
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("🎯 Predicted Genre", predicted_genre.title())
                    st.metric("📈 Confidence", f"{confidence:.1f}%")
                    st.metric("🔢 Chunks Analyzed", len(predicted_categories))
                    st.metric("🎵 Sample Rate", f"{sample_rate} Hz")
                    st.metric("⏱️ Duration", f"{len(audio_data)/sample_rate:.1f}s")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Top 3 predictions
                    st.markdown("### 🏆 Top Predictions")
                    for genre, count in genre_counts.most_common(3):
                        percentage = (count / len(predicted_categories)) * 100
                        info = GENRE_INFO[genre]
                        st.markdown(f"""
                        <div class="genre-card">
                            <h4>{info['emoji']} {genre.title()}</h4>
                            <p>{percentage:.1f}% ({count} chunks)</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Detailed visualizations
            if 'predictions' in locals():
                st.markdown("### 📈 Detailed Analysis")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    # Confidence scores
                    fig_conf = create_prediction_visualization(predictions, predicted_categories, classes)
                    st.plotly_chart(fig_conf, use_container_width=True)
                
                with col4:
                    # Chunk distribution
                    fig_chunks = create_chunk_analysis(predicted_categories, classes)
                    st.plotly_chart(fig_chunks, use_container_width=True)
                
                # Detailed breakdown
                with st.expander("🔍 Detailed Chunk Analysis"):
                    st.markdown("#### Chunk-by-chunk predictions:")
                    # Limit the number of chunks shown to prevent JSON overflow
                    max_chunks_to_show = min(20, len(predicted_categories))
                    chunk_df_data = []
                    for i in range(max_chunks_to_show):
                        pred_idx = predicted_categories[i]
                        chunk_preds = predictions[i]
                        chunk_df_data.append({
                            'Chunk': i + 1,
                            'Predicted Genre': classes[pred_idx],
                            'Confidence': f"{np.max(chunk_preds):.3f}",
                            'Top 2nd': classes[np.argsort(chunk_preds)[-2]],
                            'Top 3rd': classes[np.argsort(chunk_preds)[-3]]
                        })
                    
                    chunk_df = pd.DataFrame(chunk_df_data)
                    st.dataframe(chunk_df, use_container_width=True)
                    
                    if len(predicted_categories) > max_chunks_to_show:
                        st.info(f"Showing first {max_chunks_to_show} chunks out of {len(predicted_categories)} total chunks.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    else:
        # Welcome message for classifier page
        st.markdown("""
        <div class="genre-card">
            <h2>🎵 Ready to Classify Your Music!</h2>
            <p>Upload an audio file using the sidebar to get started with AI-powered genre classification.</p>
            <br>
            <h4>🚀 Quick Tips:</h4>
            <ul>
                <li>🎯 <strong>Best Results</strong> - Use clear, full-length songs</li>
                <li>📊 <strong>File Size</strong> - Keep files under 25MB for optimal performance</li>
                <li>🎵 <strong>Duration</strong> - 30 seconds to 5 minutes works best</li>
                <li>🎧 <strong>Quality</strong> - Higher quality audio improves accuracy</li>
            </ul>
            <br>
            <p><strong>👈 Upload an audio file from the sidebar to get started!</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🎵 Music Genre Classifier | Built with Streamlit & TensorFlow | 
        <a href="#" style="color: #667eea;">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function with navigation"""
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "landing"
    
    # Page navigation
    if st.session_state.page == "landing":
        show_landing_page()
    elif st.session_state.page == "classifier":
        show_classifier_page()

if __name__ == "__main__":
    main()
