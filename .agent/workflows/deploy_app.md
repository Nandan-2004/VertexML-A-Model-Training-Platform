---
description: How to deploy the Streamlit application
---

This guide covers two common ways to deploy your Streamlit application: Streamlit Community Cloud (easiest) and Docker (most flexible).

## Option 1: Streamlit Community Cloud
Best for: Quick demos, personal projects, and testing. Free.

1. **Push to GitHub**
   - Ensure your project is initialized as a git repository (`git init`).
   - Create a `.gitignore` file (if not exists) and add `venv/`, `.env`, and `*.pyc`.
   - Commit and push your code to a public or private GitHub repository.
   - *Note: Ensure `requirements.txt` is in the root.*

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
   - Click **"New app"**.
   - Select your repository, branch (usually `main`), and main file path (e.g., `app/main.py`).
   - Click **"Deploy!"**.

3. **Secrets Management**
   - If you use API keys (like OpenAI), go to your deployed app's **Settings** > **Secrets**.
   - Paste the contents of your local `.streamlit/secrets.toml` there.

## Option 2: Docker Container
Best for: Production, custom environments, Google Cloud Run, AWS, Azure.

1. **Create a Dockerfile**
   - Create a file named `Dockerfile` in the root directory `d:\AutoML`.
   - Use the following content:
     ```dockerfile
     # Use an official Python runtime as a parent image
     FROM python:3.9-slim

     # Set the working directory
     WORKDIR /app

     # Copy requirements and install dependencies
     COPY requirements.txt .
     RUN pip install --no-cache-dir -r requirements.txt

     # Copy the rest of the application code
     COPY . .

     # Expose the port Streamlit runs on
     EXPOSE 8501

     # Run the application
     CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
     ```

2. **Build the Image**
   Run this terminal command in `d:\AutoML`:
   ```powershell
   docker build -t automl-platform .
   ```

3. **Run Locally (to test)**
   ```powershell
   docker run -p 8501:8501 automl-platform
   ```
   Access at `http://localhost:8501`.

4. **Deploy to Cloud (e.g., Google Cloud Run)**
   - Tag and push the image to a container registry (GCR, Docker Hub).
   - Deploy the container to a service like Cloud Run, App Engine, or AWS Fargate.
