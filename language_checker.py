import streamlit as st
import pandas as pd
import re
import unicodedata
from sentence_transformers import SentenceTransformer, util
import os
import tempfile
import time
from typing import List, Tuple

# ---------- Config ----------
ENGLISH_HEADERS_IN_ORDER = ["Question", "Option A", "Option B", "Option C", "Option D"]
MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 128
CHUNK_SIZE = 1000

# Custom CSS for styling
st.markdown("""
<style>
body {
    font-family: 'Arial', sans-serif;
    background-color: #f5f5f5;
}
.stApp {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}
.main-header {
    font-size: 2.5em;
    color: #1E3A8A;
    text-align: center;
    margin-bottom: 10px;
}
.sub-header {
    font-size: 1.2em;
    color: #4B5563;
    text-align: center;
    margin-bottom: 20px;
}
.stButton>button {
    background-color: #1E3A8A;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 1em;
    border: none;
}
.stButton>button:hover {
    background-color: #3B82F6;
    color: white;
}
.stDataFrame {
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    background-color: white;
    padding: 10px;
}
.stSpinner {
    display: flex;
    justify-content: center;
    align-items: center;
}
.progress-text {
    font-size: 1em;
    color: #4B5563;
    text-align: center;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# Load model once globally for efficiency
@st.cache_resource
def load_model():
    print("Loading SentenceTransformer model...")
    start_time = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    return model

model = load_model()

# ---------- Normalization helpers ----------
_ws_re = re.compile(r"\s+")

def normalize_eng(s: str) -> str:
    """Normalize English text: NFC + lowercase + collapse spaces"""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = unicodedata.normalize("NFC", s)
    s = _ws_re.sub(" ", s.strip())
    return s.lower()

def normalize_tgt(s: str) -> str:
    """Normalize translated text: NFC + collapse spaces, NO lowercase"""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = unicodedata.normalize("NFC", s)
    s = _ws_re.sub(" ", s.strip())
    return s

def find_pairs_by_position(df: pd.DataFrame) -> List[Tuple[str, str, int, int]]:
    """Find (English column -> Translation column) pairs by position."""
    cols = list(df.columns)
    pairs = []
    for i, col in enumerate(cols):
        if col in ENGLISH_HEADERS_IN_ORDER and i + 1 < len(cols):
            pairs.append((col, cols[i + 1], i, i + 1))

    if not pairs:
        raise ValueError(
            "No (English -> Translation) pairs inferred. "
            "Ensure each English column is immediately followed by its translation."
        )
    pairs.sort(key=lambda x: x[3])
    print(f"Detected column pairs: {pairs}")
    return pairs

def cosine_similarity_percent(src_texts: List[str], tgt_texts: List[str]) -> List[float]:
    """Compute semantic similarity using the loaded model."""
    start_time = time.time()
    src_norm = [normalize_eng(x) for x in src_texts]
    tgt_norm = [normalize_tgt(x) for x in tgt_texts]

    src_enc = [f"query: {x}" for x in src_norm]
    tgt_enc = [f"query: {x}" for x in tgt_norm]

    a = model.encode(src_enc, convert_to_tensor=True, normalize_embeddings=True, batch_size=BATCH_SIZE)
    b = model.encode(tgt_enc, convert_to_tensor=True, normalize_embeddings=True, batch_size=BATCH_SIZE)

    sims = util.cos_sim(a, b).diagonal()
    sims = sims.clamp(min=0.0, max=1.0) * 100.0
    sims_list = [round(float(x), 2) for x in sims]
    print(f"Similarity computation took {time.time() - start_time:.2f} seconds for {len(src_texts)} items. First 5 scores: {sims_list[:5]}")
    return sims_list

def process_file(df: pd.DataFrame, output_path: str):
    """Process the DataFrame in chunks and save to Excel."""
    start_time = time.time()
    print(f"Input columns: {list(df.columns)}")

    pairs = find_pairs_by_position(df)

    shift = 0
    for eng_col, tr_col, eng_idx, tr_idx in pairs:
        sims_pct_all = []
        for start in range(0, len(df), CHUNK_SIZE):
            chunk_end = min(start + CHUNK_SIZE, len(df))
            src_chunk = df[eng_col].iloc[start:chunk_end].astype(str).fillna("").tolist()
            tgt_chunk = df[tr_col].iloc[start:chunk_end].astype(str).fillna("").tolist()

            chunk_start_time = time.time()
            sims_pct = cosine_similarity_percent(src_chunk, tgt_chunk)
            sims_pct_all.extend(sims_pct)
            print(f"Processed chunk {start}-{chunk_end} for {eng_col}->{tr_col} in {time.time() - chunk_start_time:.2f} seconds")

        new_col_name = f"{tr_col} Similarity (%)"
        insert_at = tr_idx + 1 + shift
        df.insert(insert_at, new_col_name, sims_pct_all)
        shift += 1
        print(f"Inserted {new_col_name} at position {insert_at}")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    print(f"Total processing and save took {time.time() - start_time:.2f} seconds. Output: {output_path}")

# Streamlit UI
st.markdown("<div class='main-header'>Language Similarity Checker</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Upload an Excel file to compare English and translated text similarity.</div>", unsafe_allow_html=True)

with st.container():
    st.write("**Instructions**: Upload an Excel file with columns 'Question', 'Option A', 'Option B', 'Option C', 'Option D' followed by their translations (e.g., 'Translated', 'Translated.1'). The app will compute similarity scores and add them as new columns.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type=["xlsx"], help="File should contain English and translated columns in pairs.")
    
    if uploaded_file:
        try:
            with st.spinner("Processing your file..."):
                start_time = time.time()
                # Progress bar
                progress_bar = st.progress(0)
                progress_text = st.markdown("<div class='progress-text'>Starting...</div>", unsafe_allow_html=True)

                # Save uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Read and process
                progress_text.markdown("<div class='progress-text'>Reading Excel file...</div>", unsafe_allow_html=True)
                df = pd.read_excel(tmp_path, sheet_name=0)
                progress_bar.progress(20)

                output_path = os.path.join(tempfile.gettempdir(), f"processed_{uploaded_file.name}")
                progress_text.markdown("<div class='progress-text'>Computing similarities...</div>", unsafe_allow_html=True)
                process_file(df, output_path)
                progress_bar.progress(80)

                # Display results
                progress_text.markdown("<div class='progress-text'>Preparing output...</div>", unsafe_allow_html=True)
                st.markdown("### Processed Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                progress_bar.progress(100)
                progress_text.markdown("<div class='progress-text'>Done!</div>", unsafe_allow_html=True)

                # Download button
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="Download Processed File",
                        data=f,
                        file_name=f"processed_{uploaded_file.name}",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download the Excel file with similarity scores."
                    )

                # Cleanup
                os.remove(tmp_path)
                print(f"Total request took {time.time() - start_time:.2f} seconds")
                st.success("Processing complete! Download the file above.")

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
            st.error(f"Processing failed: {str(e)}")
            progress_bar.progress(0)
            progress_text.markdown("<div class='progress-text'>Error occurred.</div>", unsafe_allow_html=True)

st.markdown("<hr><div style='text-align: center; color: #6B7280; font-size: 0.9em;'>Powered by Streamlit & Sentence Transformers</div>", unsafe_allow_html=True)