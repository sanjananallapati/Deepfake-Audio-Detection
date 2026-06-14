"""
Streamlit web app for deepfake audio detection.

Run locally:
    streamlit run app/streamlit_app.py

Deploy (Streamlit Community Cloud):
    Push the repo to GitHub, create a new app pointing at
    ``app/streamlit_app.py``, and make sure ``models/best_model.pt`` is in the
    repo (via Git LFS) or downloaded at startup — see the README.

The app accepts an audio file, plays it back, runs the trained model, and shows
whether the clip is **Genuine (Human)** or **Deepfake (AI-Generated)** together
with a confidence score and the log-mel spectrogram the model actually sees.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

# Make ``src`` importable when run as ``streamlit run app/streamlit_app.py``.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import audio_path_to_feature  # noqa: E402
from src.predict import load_bundle, predict_array  # noqa: E402

DEFAULT_MODEL = str(ROOT / "models" / "best_model.pt")
ACCEPTED = ["wav", "flac", "mp3", "ogg", "m4a", "aac"]

st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️", layout="centered")


@st.cache_resource(show_spinner=True)
def get_bundle(model_path: str):
    """Load the model once and cache it across reruns/sessions."""
    return load_bundle(model_path)


def mel_figure(feat: np.ndarray, sr: int, hop: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(feat, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, feat.shape[1] * hop / sr, 0, feat.shape[0]])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    ax.set_title("Log-mel spectrogram (model input)")
    fig.colorbar(im, ax=ax, format="%+.0f dB", fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def main():
    st.title("🎙️ Deepfake Audio Detector")
    st.caption("Upload a speech clip — the model classifies it as **Genuine (Human)** "
               "or **Deepfake (AI-Generated)** and reports its confidence.")

    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input("Model checkpoint path", value=DEFAULT_MODEL)
        st.markdown(
            "Label convention: **0 = Genuine**, **1 = Deepfake**. "
            "The score is the model's deepfake probability."
        )

    # Load model (with a friendly message if it is missing).
    try:
        bundle = get_bundle(model_path)
    except FileNotFoundError:
        st.error(
            f"No trained model found at `{model_path}`.\n\n"
            "Train one with `python -m src.train`, or update the path in the sidebar."
        )
        st.stop()
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to load the model: {e}")
        st.stop()

    if bundle.meta.get("val_eer") is not None:
        st.sidebar.success(
            f"Model loaded\nval EER: {bundle.meta['val_eer']*100:.2f}%\n"
            f"val acc: {bundle.meta['val_accuracy']*100:.2f}%"
        )

    uploaded = st.file_uploader("Choose an audio file", type=ACCEPTED)
    if uploaded is None:
        st.info("Awaiting an audio file…")
        return

    # Persist to a temp file so librosa can read any supported format.
    suffix = Path(uploaded.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    st.audio(uploaded, format=f"audio/{suffix.lstrip('.')}")

    with st.spinner("Analysing…"):
        try:
            feat = audio_path_to_feature(tmp_path, bundle.audio_cfg, bundle.feat_cfg)
            result = predict_array(bundle, feat)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not process this file: {e}")
            return

    label = result["label_name"]
    conf = result["confidence"] * 100
    p_fake = result["p_deepfake"]

    st.subheader("Result")
    if result["label"] == 0:
        st.success(f"### ✅ Genuine (Human)\nConfidence: **{conf:.1f}%**")
    else:
        st.error(f"### ⚠️ Deepfake (AI-Generated)\nConfidence: **{conf:.1f}%**")

    col1, col2 = st.columns(2)
    col1.metric("P(Genuine)", f"{result['p_genuine']*100:.1f}%")
    col2.metric("P(Deepfake)", f"{result['p_deepfake']*100:.1f}%")

    st.progress(min(max(p_fake, 0.0), 1.0),
                text=f"Deepfake probability: {p_fake*100:.1f}%")

    with st.expander("Show the spectrogram the model analysed"):
        fig = mel_figure(feat, bundle.audio_cfg.sample_rate, bundle.feat_cfg.hop_length)
        st.pyplot(fig)

    st.caption(
        "This is a statistical detector and can make mistakes, especially on "
        "audio unlike its training data (new synthesis methods, heavy compression, "
        "background noise). Treat the result as one signal, not proof."
    )


if __name__ == "__main__":
    main()
