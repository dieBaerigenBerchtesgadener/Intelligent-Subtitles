# Intelligent Subtitles 🎬

A machine learning-driven system for dynamically adaptive subtitle display based on audio and textual complexity analysis.

## Project Overview

Intelligent Subtitles aims to solve a common problem when watching content in a foreign language: the trade-off between subtitle visibility and visual immersion. Instead of displaying all subtitles or none at all, this system dynamically decides when subtitles are necessary based on:

- **Audio complexity**: How clear is the speech?
- **Word complexity**: How difficult are the individual words?
- **Sentence complexity**: How complex is the overall sentence structure?
- **Word importance**: How essential is each word to understanding the context?
- **Word occurrence**: How frequently has this word appeared in the video?
- **Speech rate**: How quickly are words being spoken?
- **Speech volume**: How loud is the speech compared to average?
- **Ambient volume**: How much background noise competes with speech?

## Features

- **Adaptive subtitle display**: Shows subtitles only when they're likely needed
- **Multi-language support**: English with German translations
- **Neural network-based decision making**: Trained on user preferences
- **Audio isolation**: Separates speech from background noise for better analysis
- **Real-time processing**: Works with existing subtitle files

## Technical Implementation

The pipeline consists of several key components:

1. **Preprocessing**: Loads and cleans subtitle files
2. **Audio analysis**: 
   - Extracts speech uncertainty using Whisper ASR
   - Measures speech rate (syllables/second)
   - Analyzes speech volume and ambient noise ratio
3. **Text analysis**:
   - Evaluates word complexity using CEFR levels
   - Measures sentence entropy using ModernBERT
   - Calculates word importance via masked prediction
   - Tracks word occurrence frequency
4. **Neural network model**:
   - Three-layer architecture (64-128-64 neurons)
   - Trained with F2-score optimization (recall emphasis)
   - Optional bias parameter for fine-tuning display ratio

## Model Performance

Current model metrics:
- F1-Score: 0.3825
- F2-Score: 0.5867
- Balanced Accuracy: 0.8220
- ROC-AUC: 0.8889

The model demonstrates high recall (0.78) on subtitle display decisions, ensuring important subtitles are rarely missed, with improving precision (0.30) to reduce unnecessary displays.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- CUDA-compatible GPU (recommended)

### Installation

git clone https://github.com/dieBaerigenBerchtesgadener/IntelligentSubtitles.git
cd IntelligentSubtitles
pip install -r requirements.txt

### Usage

Example usage
python main.py --video path/to/video.mp4 --subtitles path/to/subtitles.srt --bias 0.0

## Project Structure

- `audio_complexity.py`: Audio analysis features
- `feature_extraction.py`: Feature extraction pipeline
- `filter_in_out.py`: Subtitle filtering utilities
- `model.py`: Neural network implementation
- `preprocessing.py`: Data preprocessing
- `translation.py`: Subtitle translation functions
- `subtitle_generation.py`: Subtitle output formatting
- `utils.py`: Helper functions

## Future Work

- Improve precision for class 1 (display subtitles)
- Optimize pipeline for speed and robustness
- Expand training dataset with diverse audio and text comprehension levels
- Develop personalization for individual users

## Author

- Kilian Kienast

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was developed as part of a Jugend forscht research project.
- Special thanks to all who contributed to labeling the training data.