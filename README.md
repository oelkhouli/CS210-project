# CIFAR Metadata Manager

**Efficient, metadata-driven management of CIFAR-10 for computer vision.**

## Structure

- **data/raw/train/**: PNGs by class  
- **src/**: ingestion, metadata, storage, indexing, query, dataloader  
- **scripts/**: preview, Streamlit UI, training  

## Quickstart

1. Create & activate venv  
 
 python -m venv venv // to download venv into folder

   .\venv\Scripts\activate.bat // on Windows cmd to activate venv

   .\venv\Scripts\Activate.ps1 // on Windows for Powershell
   
   // If you get an execution‑policy error, you can temporarily bypass it with:
   
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   // Then enter:
   .\venv\Scripts\Activate.ps1
   
2. Install dependencies

   pip install -r requirements.txt

3. Ingest & index CIFAR-10

   python src/ingestion.py --root_dir data/raw
   python -m src.indexer --root_dir data\raw\train --db_path cifar_meta.db

4. Preview a filtered subset

ex:  python scripts/preview.py --min-bright 80 --max-entropy 5.0 --limit 9

5. Run the interactive UI

   streamlit run scripts/streamlit_app.py

6. Train on Filtered Data

ex: python scripts/train_filtered.py --min-bright 80 --max-entropy 5.0  --min-contrast 20.0  --min-sharpness 100.0  --limit 500 --epochs 5 --device cuda

