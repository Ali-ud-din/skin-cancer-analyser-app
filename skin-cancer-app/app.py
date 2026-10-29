"""
SkinAI | Intelligent Skin Lesion Analysis
=========================================
A Streamlit dashboard for a skin-cancer CNN classifier, modelled on the
LungAI reference app (dark theme, sidebar navigation, analyzer + history
+ about + resources).

BEFORE DEPLOYING: fill in the CONFIG section below from the training
notebook (class names and their order, input size, preprocessing,
dataset size, reported accuracy, resource links).
"""

import io
import os
import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# =====================================================================
# CONFIG  --  EDIT THESE FROM THE TRAINING NOTEBOOK
# =====================================================================
APP_NAME = "SkinAI"
APP_TAGLINE = "Intelligent Skin Lesion Analysis"
MODEL_PATH = "skin_cancer_model.keras"  # the model exported from the notebook
# Resolve the model path relative to this file, not the working directory.
if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_PATH)
IMG_SIZE = (128, 128)              # (height, width) the model was trained on
PREPROCESS = "rescale"             # rescale | mobilenetv2 | efficientnet | none

# Class labels IN THE MODEL'S OUTPUT ORDER.
# !!! CONFIRM BEFORE DEPLOYING: image_dataset_from_directory assigns
# indices ALPHABETICALLY by subfolder name. Run
#     print(train_dataset.class_names)
# in the notebook and put the two folder names here, in that order.
CLASS_NAMES = [
    "Class 0",   # minority class in the notebook (261 test images)
    "Class 1",   # majority class in the notebook (781 test images)
]
CLASS_INFO = {
    "Class 0": ("Class 0",
                "Minority class of the training dataset (folder name to be confirmed)."),
    "Class 1": ("Class 1",
                "Majority class of the training dataset (folder name to be confirmed)."),
}
# Classes to highlight with the red warning card once names are known
# (e.g. {"malignant"}). Leave empty while labels are unconfirmed.
MALIGNANT = set()

DATASET_NAME = "Custom skin lesion dataset (Google Drive)"
DATASET_IMAGES = "3,506"
REPORTED_ACCURACY = "~77%"
# Per-class counts in the notebook's 30% TEST split (1,042 images).
CLASS_DISTRIBUTION = {
    "Class 0": 261, "Class 1": 781,
}
# Classification report from the notebook (test split).
CLASS_REPORT = {
    "Class 0": {"Precision": 0.55, "Recall": 0.45, "F1-score": 0.49},
    "Class 1": {"Precision": 0.83, "Recall": 0.88, "F1-score": 0.85},
}
# Confusion matrix (rows = true, cols = predicted).
CONFUSION = [[117, 144], [96, 685]]
MODEL_PARAMS = {                   # from the notebook
    "Framework": "TensorFlow / Keras",
    "Architecture": "Custom CNN (Conv 32/64/128 + Dense 256)",
    "Optimizer": "Adam",
    "Loss Function": "Categorical Cross-Entropy",
    "Epochs": "100 + 50",
    "Batch Size": "32",
    "Train / Test Split": "70% / 30%",
    "Total Parameters": "6,516,546 (~24.9 MB)",
    "Input Size": f"{IMG_SIZE[0]} x {IMG_SIZE[1]}",
    "Preprocessing": "Rescale 1/255",
    "Output Activation": "Softmax (2 classes)",
}
LINKS = {                          # EDIT: project resource links
    "Dataset": "https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000",
    "Google Colab Notebook": "https://colab.research.google.com/drive/1Yz2CwiyS4AsCfr5v39hK7s5UTY1s9HwV?usp=drive_link",
    "GitHub Repository": "#",
    "Live Deployment": "#",
    "Project Report": "#",
    "Project Presentation": "#",
    "LinkedIn Project Post": "#",
    "YouTube Demonstration": "#",
}
# =====================================================================

st.set_page_config(
    page_title=f"{APP_NAME} | {APP_TAGLINE} - Streamlit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
  .stApp { background-color: #0b1220; }
  .hero {
    background: linear-gradient(120deg, #062a33 0%, #0a4a54 55%, #0e7c86 100%);
    border-radius: 18px; padding: 34px 38px; color: #eafcff;
    position: relative; overflow: hidden; margin-bottom: 8px;
  }
  .hero::after {
    content: ""; position: absolute; right: -70px; top: -90px;
    width: 320px; height: 320px; border: 2px solid rgba(255,255,255,.14);
    border-radius: 50%;
    box-shadow: 0 0 0 46px rgba(255,255,255,.05), 0 0 0 110px rgba(255,255,255,.03);
  }
  .hero .eyebrow { color: #57e6c9; letter-spacing: 3px; font-size: 12px; font-weight: 700; }
  .hero h1 { color: #ffffff; font-size: 46px; margin: 6px 0 2px 0; }
  .hero h3 { color: #bdf3ea; font-weight: 400; margin: 0 0 12px 0; }
  .hero p { color: #d7f5f0; max-width: 900px; }
  .pill {
    display: inline-block; border: 1px solid rgba(255,255,255,.45);
    color: #eafcff; border-radius: 999px; padding: 3px 14px;
    font-size: 13px; margin: 8px 8px 0 0;
  }
  .eyebrow2 { color: #57e6c9; letter-spacing: 3px; font-size: 12px; font-weight: 700; }
  .metric-card {
    background: #111a2e; border: 1px solid #1e2c47; border-radius: 14px;
    padding: 18px 20px; text-align: left;
  }
  .metric-card .v { font-size: 30px; font-weight: 700; color: #ffffff; }
  .metric-card .k { color: #9fb4cc; font-size: 14px; }
  .result-card {
    border-radius: 14px; padding: 22px 24px; color: #ffffff; margin: 10px 0;
  }
  .result-bad { background: linear-gradient(120deg, #571f1f, #8c2f2f); }
  .result-good { background: linear-gradient(120deg, #0d4d33, #12805a); }
  .footer { color: #7d90a8; text-align: center; font-size: 13px; margin-top: 42px; }
  .disc { color: #ffb4a2; font-size: 14px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def hero():
    pills = "".join(
        f'<span class="pill">{p}</span>'
        for p in ["Custom CNN", "TensorFlow / Keras",
                  f"{len(CLASS_NAMES)}-Class Classification", "Academic Research"]
    )
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">DEEP LEARNING RESEARCH PLATFORM</div>
          <h1>{APP_NAME}</h1>
          <h3>{APP_TAGLINE}</h3>
          <p>A research-focused medical imaging platform powered by a
          Convolutional Neural Network. Analyze skin lesion images, explore
          classification results, and access complete project documentation
          through one integrated dashboard.</p>
          {pills}
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        f'<div class="footer">{APP_NAME} | {APP_TAGLINE}<br>'
        "Deep Learning Semester Project<br>"
        "Built with Python, TensorFlow, Keras and Streamlit<br>"
        "Educational and Research Use Only</div>",
        unsafe_allow_html=True,
    )


def metric_cards():
    c1, c2, c3 = st.columns(3)
    cards = [
        (DATASET_IMAGES, "Dataset Images"),
        (str(len(CLASS_NAMES)), "Classification Classes"),
        (REPORTED_ACCURACY, "Reported Test Accuracy"),
    ]
    for col, (v, k) in zip((c1, c2, c3), cards):
        col.markdown(
            f'<div class="metric-card"><div class="v">{v}</div>'
            f'<div class="k">{k}</div></div>',
            unsafe_allow_html=True,
        )
    st.caption("Accuracy reported in the original notebook. "
               "The dataset splitting procedure requires independent validation.")


# ---------------------------------------------------------------------
# Model loading + prediction
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the trained model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file '{MODEL_PATH}' not found next to app.py."
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model, None
    except Exception as e:  # pragma: no cover
        return None, f"Could not load the model: {e}"


def preprocess(image: Image.Image) -> np.ndarray:
    """Mirror the notebook preprocessing: resize, drop alpha, scale."""
    img = image.convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0]))
    arr = np.asarray(img).astype("float32")
    if PREPROCESS == "rescale":
        arr = arr / 255.0
    elif PREPROCESS == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        arr = preprocess_input(arr)
    elif PREPROCESS == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def make_pdf_report(record: dict) -> bytes:
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 18)
    pdf.cell(0, 12, f"{APP_NAME} - Skin Lesion Analysis Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 8, f"Date: {record['time']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Image: {record['filename']}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 9, f"Predicted class: {record['label']}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 8, f"Confidence: {record['confidence']:.2%}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 9, "Class scores", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 11)
    for name, prob in record["probs"]:
        pdf.cell(0, 7, f"{name}: {prob:.2%}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font("helvetica", "I", 9)
    pdf.multi_cell(0, 5, "Research prototype. Not intended for clinical "
                         "diagnosis. This project has not been clinically validated.")
    return bytes(pdf.output())


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"## {APP_NAME}")
    st.caption(APP_TAGLINE.upper())
    st.markdown("### Dashboard Navigation")
    page = st.radio(
        "Dashboard Navigation",
        ["Scan Analyzer", "Analysis History", f"About {APP_NAME}", "Project Resources"],
        label_visibility="collapsed",
    )
    st.markdown("### Model Configuration")
    st.markdown(
        f"- **Architecture:** {MODEL_PARAMS['Architecture']}\n"
        f"- **Framework:** {MODEL_PARAMS['Framework']}\n"
        f"- **Classes:** {len(CLASS_NAMES)}\n"
        f"- **Input Processing:** {IMG_SIZE[0]} x {IMG_SIZE[1]}\n"
        f"- **Preprocessing:** {PREPROCESS}"
    )
    st.markdown("### Quick Access")
    for label, url in LINKS.items():
        st.link_button(label, url, use_container_width=True)
    st.markdown("---")
    st.markdown('<p class="disc">Research prototype. Not intended for '
                "clinical diagnosis.</p>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

hero()

# ---------------------------------------------------------------------
# 1. Scan Analyzer
# ---------------------------------------------------------------------
if page == "Scan Analyzer":
    st.markdown('<div class="eyebrow2">IMAGE CLASSIFICATION</div>', unsafe_allow_html=True)
    st.header("Skin Lesion Analyzer")
    st.write("Upload a skin lesion image to analyze it using the trained CNN model.")
    metric_cards()

    model, err = load_model()
    if err:
        st.warning(f"{err} Add the trained model file and reload the app.")

    uploaded = st.file_uploader("Upload Skin Lesion Image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload a skin lesion image to begin analysis.")
    else:
        image = Image.open(uploaded)
        col_img, col_res = st.columns([1, 1])
        with col_img:
            st.image(image, caption=uploaded.name, use_container_width=True)
        with col_res:
            if st.button("Analyze Image", type="primary", disabled=model is None):
                with st.spinner("Analyzing..."):
                    probs = model.predict(preprocess(image), verbose=0)[0]
                order = np.argsort(probs)[::-1]
                top = int(order[0])
                code = CLASS_NAMES[top]
                full_name, desc = CLASS_INFO.get(code, (code, ""))
                conf = float(probs[top])
                record = {
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": uploaded.name,
                    "code": code,
                    "label": full_name,
                    "confidence": conf,
                    "probs": [(CLASS_NAMES[i], float(probs[i])) for i in order],
                    "image": image.copy(),
                }
                st.session_state.history.append(record)
                st.session_state.last = record

            record = st.session_state.get("last")
            if record and record["filename"] == uploaded.name:
                style = "result-bad" if record["code"] in MALIGNANT else "result-good"
                st.markdown(
                    f'<div class="result-card {style}">'
                    f"<h3 style='margin:0'>Prediction: {record['label']}</h3>"
                    f"<p style='margin:6px 0 0 0'>Confidence: "
                    f"<b>{record['confidence']:.2%}</b></p></div>",
                    unsafe_allow_html=True,
                )
                st.write(CLASS_INFO.get(record["code"], ("", ""))[1])
                st.subheader("Classification Scores")
                df = pd.DataFrame(record["probs"], columns=["Class", "Probability"])
                st.bar_chart(df.set_index("Class"))
                st.dataframe(df.assign(Probability=df["Probability"].map(lambda p: f"{p:.2%}")),
                             use_container_width=True, hide_index=True)
                st.download_button(
                    "Download PDF Report",
                    data=make_pdf_report(record),
                    file_name="skinai_report.pdf",
                    mime="application/pdf",
                )
    st.markdown('<p class="disc">SkinAI is an educational research prototype. '
                "Do not use its results for medical diagnosis or treatment "
                "decisions.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 2. Analysis History
# ---------------------------------------------------------------------
elif page == "Analysis History":
    st.markdown('<div class="eyebrow2">SESSION ACTIVITY</div>', unsafe_allow_html=True)
    st.header("Analysis History")
    st.write("Review predictions generated during your current application session.")
    if not st.session_state.history:
        st.info("No images have been analyzed yet.")
    else:
        for rec in reversed(st.session_state.history):
            c1, c2, c3 = st.columns([1, 3, 2])
            c1.image(rec["image"], width=110)
            c2.markdown(f"**{rec['label']}**  \n{rec['filename']}  \n{rec['time']}")
            c3.metric("Confidence", f"{rec['confidence']:.2%}")
            st.markdown("---")
    st.caption("Analysis history is stored only in the current Streamlit session.")

# ---------------------------------------------------------------------
# 3. About
# ---------------------------------------------------------------------
elif page == f"About {APP_NAME}":
    st.markdown('<div class="eyebrow2">PROJECT OVERVIEW</div>', unsafe_allow_html=True)
    st.header(f"About {APP_NAME}")
    st.write(f"{APP_NAME} is a Deep Learning semester project designed to "
             "classify skin lesion images using a Convolutional Neural "
             "Network developed with TensorFlow and Keras.")
    metric_cards()

    st.subheader("Problem Statement")
    st.write("Skin lesion images contain complex visual patterns; this "
             "project explores CNN-based feature learning to distinguish "
             f"{len(CLASS_NAMES)} skin lesion categories from dermoscopic "
             "images, supporting early screening research.")

    st.subheader("Dataset Information")
    st.write(f"**Dataset:** {DATASET_NAME} - {DATASET_IMAGES} images, "
             f"{len(CLASS_NAMES)} classes. Per-class counts below are for "
             "the notebook's 30% test split (1,042 images).")
    df = pd.DataFrame(
        [{"Class": CLASS_INFO.get(k, (k,))[0], "Images": v}
         for k, v in CLASS_DISTRIBUTION.items()]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("Class"))

    st.subheader("CNN Architecture")
    st.write("The model is a Convolutional Neural Network trained with "
             "TensorFlow/Keras (see the training notebook for the exact "
             "layer-by-layer architecture). Key training parameters:")
    st.table(pd.DataFrame(MODEL_PARAMS.items(), columns=["Parameter", "Value"]))

    st.subheader("Model Processing Pipeline")
    steps = [
        "Upload a skin lesion image.",
        f"Resize the image to {IMG_SIZE[0]} x {IMG_SIZE[1]} pixels.",
        "Convert the image into a NumPy array.",
        "Remove the alpha channel if present (convert to RGB).",
        f"Apply preprocessing ({PREPROCESS}).",
        "Run inference using the trained CNN.",
        "Identify the highest-scoring class.",
        "Display the classification scores.",
        "Generate a PDF analysis report.",
    ]
    for i, s in enumerate(steps, 1):
        st.markdown(f"**{i}.** {s}")

    st.subheader("Classification Report (test split)")
    rep_df = pd.DataFrame(CLASS_REPORT).T.reset_index(names="Class")
    st.dataframe(rep_df, use_container_width=True, hide_index=True)
    st.subheader("Confusion Matrix (test split)")
    cm = pd.DataFrame(CONFUSION,
                      index=[f"True {c}" for c in CLASS_NAMES],
                      columns=[f"Pred {c}" for c in CLASS_NAMES])
    st.table(cm)

    st.subheader("Reported Model Results")
    st.write(f"The original notebook reports a test accuracy of "
             f"**{REPORTED_ACCURACY}**. Accuracy reported in the original "
             "notebook; the dataset splitting procedure requires "
             "independent validation.")

    st.subheader("Project Documentation")
    for label in ["Project Report", "Google Colab Notebook",
                  "Project Presentation", "GitHub Repository"]:
        st.link_button(label, LINKS.get(label, "#"))
    st.markdown('<p class="disc">This project has not been clinically '
                "validated.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 4. Project Resources
# ---------------------------------------------------------------------
elif page == "Project Resources":
    st.markdown('<div class="eyebrow2">RESEARCH MATERIALS</div>', unsafe_allow_html=True)
    st.header("Project Resources")
    st.write("Access all project materials, including the dataset, "
             "notebook, source code, deployment, report and presentation.")
    descs = {
        "Dataset": f"{DATASET_NAME} skin lesion image dataset.",
        "Google Colab Notebook": "Model architecture, training, and evaluation.",
        "GitHub Repository": "Source code and project implementation.",
        "Live Deployment": "Deployed Streamlit dashboard.",
        "Project Report": "Detailed methodology, architecture, and results.",
        "Project Presentation": "Project slides and research overview.",
        "LinkedIn Project Post": "Project announcement on LinkedIn.",
        "YouTube Demonstration": "Recorded project demonstration.",
    }
    cols = st.columns(2)
    for i, (label, url) in enumerate(LINKS.items()):
        with cols[i % 2]:
            st.markdown(f"### {label}")
            st.write(descs.get(label, ""))
            st.link_button("Open Resource", url)
            st.markdown("---")
    st.caption("Ensure that externally hosted documents are accessible "
               "to your project evaluator.")

footer()
