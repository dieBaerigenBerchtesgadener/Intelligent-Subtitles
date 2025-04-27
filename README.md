# Intelligent Subtitles

This repository contains the code for "Intelligente Untertitel: Automatische Anpassung der Untertiteldarstellung basierend auf Audio- und Textverständlichkeit" - a system that dynamically displays subtitles based on audio and text comprehensibility factors.

## Overview

The project addresses limitations of conventional subtitle systems by developing an adaptive approach that displays subtitles based on audio and text comprehensibility needs. The methodology combines nine carefully developed features with user preferences in a pipeline that uses an Extreme Gradient Boosting model for selective subtitle display.

The quantitative evaluation shows promising results with an F2-score of 0.6122, a Balanced Accuracy of 0.8064, and an ROC-AUC value of 0.8927. The model demonstrates particular strength in suppressing unnecessary subtitles (precision of 0.97 for the "Do not display" class), but shows room for improvement in precision for the "Display" class (0.37).

## Features

The system analyzes nine key features:

* Word complexity
* Sentence complexity
* Word importance
* Word occurrence
* Word familiarity
* Audio complexity
* Speaking speed
* Ambient volume
* Relative volume

## Installation

    # Clone the repository
    git clone https://github.com/dieBaerigenBerchtesgadener/Intelligent-Subtitles.git
    cd Intelligent-Subtitles
    
    # Install dependencies
    pip install -r requirements.txt

## Usage

### Web Interface

The project includes a Streamlit web interface that can be started with:

    streamlit run 🏠Home.py

### Processing Videos

To process a video with the system:

1. Place your video file and its original subtitles in the `/data` folder
2. Name your files consistently: `{name}.mp4` for the video and `{name}.srt` for the subtitles
3. Run the application through the Streamlit interface or execute the notebook

### Manual Execution

You can manually run the code, including the training of the final model and evaluation, by executing:

    jupyter notebook main.ipynb

## Note

The training dataset is not included in this repository.

## Repository Structure

* `🏠Home.py`: Main Streamlit application entry point
* `main.ipynb`: Jupyter notebook containing the complete pipeline, model training, and evaluation
* `/data`: Directory for input videos and subtitle files

## Results

The evaluation on the test set yielded:

* F2-Score: 0.6122
* Balanced Accuracy: 0.8064
* ROC-AUC: 0.8927
* Precision (Do not display class): 0.97
* Recall (Display class): 0.73
