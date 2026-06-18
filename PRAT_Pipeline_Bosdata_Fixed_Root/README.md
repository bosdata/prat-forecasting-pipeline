# PRAT Forecasting Pipeline - Fixed Root Package

This package is arranged so that `app.py` and `requirements.txt` are in the first folder you open after extracting.

## Run on Windows PowerShell

Open PowerShell inside this extracted folder, then run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Corrections included

- Optional grouping columns allow multiple selections.
- Optional ID columns allow multiple selections.
- Changing forecast view does not clear results.
- Daily, weekly, monthly, quarterly, semi-annual and annual actual/forecast views are shown.
- PDF report download is included.
- Editable LaTeX report download is included.
- Text cleaning, missing value checks and duplicate checks are included.
- Footer added: Created by Bosdata Tech Team, Brian Sifuna Obware.
