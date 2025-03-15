import streamlit as st
import pandas as pd
import os
import tempfile
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from shared import process_subtitles
from utils import cefr_levels  # Import the CEFR mapping

def main():
    st.title("Create Dataset")
    
    # Initialize session state
    if 'full_df' not in st.session_state:
        st.session_state.full_df = pd.DataFrame()
    if 'display_df' not in st.session_state:
        st.session_state.display_df = pd.DataFrame()
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None
    if 'english_level' not in st.session_state:
        st.session_state.english_level = None
    if 'audio_level' not in st.session_state:
        st.session_state.audio_level = None

    # File selection UI
    option = st.radio("Choose input method:", ["Local files", "File upload"], index=0)

    if option == "Local files":
        name = st.text_input("Video name (without extension):")
        if name:
            base_path = os.path.join("data", name)
            reference_file = f"{base_path}.srt"
            
            if os.path.exists(reference_file) and st.button("Run pipeline"):
                video_formats = [".mp4", ".mpeg", ".wav", ".mp3"]
                for fmt in video_formats:
                    if os.path.exists(f"{base_path}{fmt}"):
                        st.session_state.audio_file = f"{base_path}{fmt}"
                        break
                
                if st.session_state.audio_file:
                    df, _ = process_subtitles(name, st.session_state.audio_file, reference_file, predict=False)
                    st.session_state.full_df = df.copy()
                    st.session_state.display_df = df[["word", "display"]].copy()
                else:
                    st.error("No supported video file found")
    else:
        video_file = st.file_uploader("Upload video/audio:", type=["mp4", "mpeg", "wav", "mp3"])
        srt_file = st.file_uploader("Upload SRT file:", type=["srt"])
        
        if video_file and srt_file and st.button("Run pipeline"):
            with tempfile.NamedTemporaryFile(delete=False) as tmp_vid, \
                 tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as tmp_srt:
                
                tmp_vid.write(video_file.getvalue())
                tmp_srt.write(srt_file.getvalue())
                
                st.session_state.audio_file = tmp_vid.name
                df, _ = process_subtitles("uploaded", tmp_vid.name, tmp_srt.name, predict=False)
                st.session_state.full_df = df.copy()
                st.session_state.display_df = df[["word", "display"]].copy()

    # Display and editing UI
    if not st.session_state.display_df.empty and st.session_state.audio_file:
        col3, col4 = st.columns([1, 1])

        with col3:
            st.session_state.english_level = st.selectbox(
                "English level:",
                options=list(cefr_levels.keys()),
                index=None,
                placeholder="Select your English level",
                key='english_level_select'
            )
        
        with col4:
            st.session_state.audio_level = st.selectbox(
                "Audio level:",
                options=list(cefr_levels.keys()),
                index=None,
                placeholder="Select your Audio level",
                key='audio_level_select'
            )

        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.header("Word Labels")
            gb = GridOptionsBuilder.from_dataframe(st.session_state.display_df)
            gb.configure_column("display", editable=True, cellEditor="agCheckboxCellEditor")
            gb.configure_column("word", editable=False)
            grid_options = gb.build()
            
            grid_response = AgGrid(
                st.session_state.display_df,
                gridOptions=grid_options,
                update_mode=GridUpdateMode.VALUE_CHANGED,
                data_return_mode='FILTERED_AND_SORTED',
                fit_columns_on_grid_load=True,
                height=400,
                key='grid'
            )
            
            # Sync changes to full dataframe using word as key
            if not grid_response['data'].equals(st.session_state.display_df):
                st.session_state.display_df = grid_response['data']
                full_idx = st.session_state.full_df.set_index('word')
                display_idx = st.session_state.display_df.set_index('word')
                full_idx.update(display_idx)
                st.session_state.full_df = full_idx.reset_index()

        with col2:
            st.header("Video Player")
            st.video(st.session_state.audio_file)

        # Save functionality with level conversion
        if st.button("Save labels"):
            try:
                # Convert selected levels to numerical values
                if st.session_state.english_level:
                    st.session_state.full_df['english_level'] = cefr_levels[st.session_state.english_level]
                if st.session_state.audio_level:
                    st.session_state.full_df['audio_level'] = cefr_levels[st.session_state.audio_level]
                
                # Save to CSV
                base_name = os.path.splitext(os.path.basename(st.session_state.audio_file))[0]
                save_path = os.path.join("data", f"{base_name}_labeled.csv")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                st.session_state.full_df.to_csv(save_path, index=False)
                
                st.success(f"Labels saved to {save_path}")
            except KeyError as e:
                st.error(f"Invalid level selection: {e}")
            except Exception as e:
                st.error(f"Error saving file: {e}")

if __name__ == "__main__":
    main()
