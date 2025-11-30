#!/usr/bin/env python3
"""
Script to set up Dagshub integration for MLflow and DVC.

Run this script after creating your Dagshub repository.
"""

import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv


def setup_dagshub():
    """Configure Dagshub for MLflow and DVC."""
    load_dotenv()
    
    username = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")
    repo_name = os.getenv("DAGSHUB_REPO_NAME", "weather-prediction-mlops")
    
    if not username or not token:
        print("❌ Error: DAGSHUB_USERNAME and DAGSHUB_TOKEN must be set in .env")
        print("\nTo set up:")
        print("1. Create an account at https://dagshub.com")
        print("2. Create a new repository")
        print("3. Go to Settings > Tokens and create an access token")
        print("4. Add to your .env file:")
        print("   DAGSHUB_USERNAME=your_username")
        print("   DAGSHUB_TOKEN=your_token")
        print("   DAGSHUB_REPO_NAME=your_repo_name")
        return False
    
    print(f"🔧 Setting up Dagshub for {username}/{repo_name}")
    
    # Configure DVC remote
    dvc_url = f"https://dagshub.com/{username}/{repo_name}.dvc"
    
    print(f"\n📦 Configuring DVC remote: {dvc_url}")
    
    subprocess.run(["dvc", "remote", "add", "-d", "-f", "dagshub", dvc_url], check=True)
    subprocess.run(["dvc", "remote", "modify", "dagshub", "--local", "auth", "basic"], check=True)
    subprocess.run(["dvc", "remote", "modify", "dagshub", "--local", "user", username], check=True)
    subprocess.run(["dvc", "remote", "modify", "dagshub", "--local", "password", token], check=True)
    
    print("✅ DVC remote configured")
    
    # Configure MLflow
    mlflow_uri = f"https://dagshub.com/{username}/{repo_name}.mlflow"
    
    print(f"\n📊 MLflow tracking URI: {mlflow_uri}")
    print("   Set in your environment or code:")
    print(f'   export MLFLOW_TRACKING_URI="{mlflow_uri}"')
    print(f'   export MLFLOW_TRACKING_USERNAME="{username}"')
    print(f'   export MLFLOW_TRACKING_PASSWORD="<your_token>"')
    
    print("\n✅ Dagshub setup complete!")
    print("\n📋 Next steps:")
    print("1. Initialize git repository: git init")
    print("2. Add Dagshub as remote: git remote add origin https://dagshub.com/{}/{}.git".format(username, repo_name))
    print("3. Push your code: git push -u origin main")
    print("4. Run the training script to see experiments in Dagshub")
    
    return True


if __name__ == "__main__":
    setup_dagshub()

