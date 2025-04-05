# Create virtual environment
python -m venv .venv

# Activate the environment
 .venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install packages
pip install -r requirements.txt

#Output
api: https://hf-demo-api.up.railway.app/
gradio port: https://hf-demo-api.up.railway.app:7860