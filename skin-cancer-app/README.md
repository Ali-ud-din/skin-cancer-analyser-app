# SkinAI — Intelligent Skin Lesion Analysis

A Streamlit dashboard for a skin-cancer CNN classifier, modelled on the
LungAI lung-CT app: dark theme, teal hero banner, sidebar navigation with
**Scan Analyzer**, **Analysis History**, **About** and **Project Resources**.

## Before you run it — 3 things to finish

1. **Export your trained model from Colab.** The notebook never saves
   the model, so add this cell at the end and run it (if the training
   runtime has expired, re-run the notebook first):
   ```python
   model.save('skin_cancer_model.keras')
   from google.colab import files
   files.download('skin_cancer_model.keras')
   ```
   Put the downloaded file next to `app.py`.
2. **Confirm the class order.** In the notebook run
   `print(train_dataset.class_names)` and put the two folder names, in
   the printed (alphabetical) order, into `CLASS_NAMES` in `app.py`.
   Until then the app shows neutral "Class 0" / "Class 1" labels.
   The rest of the CONFIG (128x128 input, rescale 1/255, ~77% test
   accuracy, 3,506 images) is already filled in from the notebook.
3. **Replace the placeholder links** (`#`) in `LINKS` with your GitHub,
   report, presentation, LinkedIn and YouTube URLs as you publish them.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud (free)

1. Push this folder to a GitHub repo (app.py, requirements.txt,
   .streamlit/config.toml, model file).
   - If the model file is over GitHub's 100 MB limit, keep it on
     Hugging Face Hub or Google Drive and download it in `app.py`
     at startup instead — ask Muse to wire that up.
2. Go to share.streamlit.io → New app → pick the repo → main file
   `app.py` → Deploy.
3. Your app gets a public URL like `https://<name>.streamlit.app`.

## Notes

- Predictions run entirely with your own trained model; if the model
  file is missing the app still loads and shows a warning.
- The app is a research/education prototype — it carries the same
  "not for clinical diagnosis" disclaimers as the reference app.
