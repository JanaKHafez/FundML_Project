# FundML_Project

This repository contains code and resources for machine learning experiments and model training related to hate speech detection and user feedback analysis.

## Project Structure
- `backend_api.py`: Backend API implementation.
- `phase2.ipynb`, `phase3.ipynb`: Jupyter notebooks for different project phases.
- `phase2.py`, `phase3.py`: Python scripts for project phases.
- `requirements.txt`: Python dependencies.
- `streamlit_app.py`: Streamlit web application.
- `testGPU.py`: Script to test GPU availability.
- `trainModel.py`, `trainModel_clean.py`: Model training scripts.
- `user_feedback.csv`: Collected user feedback data.
- `Output/`: Contains model outputs, metrics, and processed datasets.
- `myenv/`: Python virtual environment.

## Setup
1. Clone the repository:
   ```powershell
   git clone https://github.com/JanaKHafez/FundML_Project.git
   ```
2. Create and activate a Python virtual environment (or use `myenv`):
   ```powershell
   python -m venv myenv
   .\myenv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage
- Run the Streamlit app:
  ```powershell
  streamlit run streamlit_app.py
  ```
- Run the backend API:
  ```powershell
  python backend_api.py
  ```
- Train model:
  ```powershell
  python trainModel.py
  ```
- Explore notebooks for analysis and experiments.

## Output
- Model metrics, trained models, and processed datasets are saved in the `Output/` directory.
