import streamlit as st
import pandas as pd
import os
import tempfile
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from utils import cefr_levels  # Import the CEFR mapping

def main():
    # Set the page to wide mode
    st.set_page_config(layout="wide", page_title="Edit Dataset")
    st.title("Edit Dataset")
    
    # Initialize session state
    if 'full_df' not in st.session_state:
        st.session_state.full_df = pd.DataFrame()
    if 'display_df' not in st.session_state:
        st.session_state.display_df = pd.DataFrame()
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None
    if 'language_level' not in st.session_state:
        st.session_state.language_level = None
    if 'audio_level' not in st.session_state:
        st.session_state.audio_level = None
    if 'creator' not in st.session_state:
        st.session_state.creator = None

    # File selection UI
    option = st.radio("Choose input method:", ["Local files", "File upload"], index=0)

    if option == "Local files":
        name = st.text_input("Video name (without extension):")
        if not name:
            st.stop()
        else:
            name = "data\\" + name
            st.session_state.name = name
            st.session_state.audio_file = f"{name}.mp4"

        audio_file = f"{name}.mp4"
        csv_file = f"{name}.csv"
            
        if os.path.exists(csv_file) and st.button("Load data") and os.path.exists(audio_file):
            try:
                df = pd.read_csv(csv_file)
                df['display'] = False
                st.session_state.full_df = df.copy()
                if 'display' in df.columns:
                    st.session_state.display_df = df[["word", "display"]].copy()
                else:
                    st.session_state.display_df = df[["word"]].copy()
                    st.session_state.display_df['display'] = True
            except Exception as e:
                st.error(f"Error loading data: {e}")
    else:
        video_file = st.file_uploader("Upload video/audio:", type=["mp4", "mpeg", "wav", "mp3"])
        csv_file = st.file_uploader("Upload CSV file:", type=["csv"])
        
        if video_file and csv_file and st.button("Load data"):
            with tempfile.NamedTemporaryFile(delete=False) as tmp_vid, \
                 tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
                
                tmp_vid.write(video_file.getvalue())
                tmp_csv.write(csv_file.getvalue())
                
                st.session_state.audio_file = tmp_vid.name
                try:
                    df = pd.read_csv(tmp_csv.name)
                    st.session_state.full_df = df.copy()
                    if 'display' in df.columns:
                        st.session_state.display_df = df[["word", "display"]].copy()
                    else:
                        st.session_state.display_df = df[["word"]].copy()
                        st.session_state.display_df['display'] = True
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")

    # Display and editing UI
    if not st.session_state.display_df.empty:
        st.session_state.creator = st.text_input("Creator:", help="Enter your name or nickname")
        col3, col4 = st.columns([1, 1])

        with col3:
            st.session_state.language_level = st.selectbox(
                "English level:",
                options=list(cefr_levels.keys()),
                index=None,
                placeholder="Select your English level",
                key='language_level_select',
                help="Not sure about your level? Take a test: https://test-english.com/level-test/"
            )
        
        with col4:
            st.session_state.audio_level = st.selectbox(
                "Audio level:",
                options=list(cefr_levels.keys()),
                index=None,
                placeholder="Select your Audio level",
                key='audio_level_select',
                help="Not sure about your listening level? Take a test: https://www.oxfordonlineenglish.com/english-level-test/listening"
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
                height=800,
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
                if st.session_state.language_level:
                    st.session_state.full_df['language_level'] = cefr_levels[st.session_state.language_level]
                if st.session_state.audio_level:
                    st.session_state.full_df['audio_level'] = cefr_levels[st.session_state.audio_level]
                
                # Save to CSV
                base_name = os.path.splitext(os.path.basename(st.session_state.audio_file))[0]
                save_path = os.path.join("data", f"{base_name}_labeled_{st.session_state.creator}.csv")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                st.session_state.full_df.to_csv(save_path, index=False)
                
                st.success(f"Labels saved to {save_path}")
            except KeyError as e:
                st.error(f"Invalid level selection: {e}")
            except Exception as e:
                st.error(f"Error saving file: {e}")

if __name__ == "__main__":
    main()
