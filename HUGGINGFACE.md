# Deploy to Hugging Face Spaces

This guide walks through deploying the Intern Performance Predictor app on Hugging Face Spaces using Docker.

---

## Prerequisites

- A [Hugging Face](https://huggingface.co/) account
- Git installed on your machine
- Your project pushed to GitHub: `https://github.com/awais-dev-ai/Internee_Performance_Predictor`

---

## Step 1: Create a Hugging Face Space

1. Log in to [Hugging Face](https://huggingface.co/login)
2. Click your profile picture → **New Space**
3. Fill in:
   - **Space Name**: `Intern-Performance-Predictor`
   - **License**: MIT (recommended)
   - **Space SDK**: Select **Docker**
   - **Docker Template**: **Blank** (we'll use our own Dockerfile)
   - **Space Hardware**: **CPU basic** (free tier)
4. Click **Create Space**

---

## Step 2: Add Your Hugging Face Remote

After creating the Space, Hugging Face gives you a Git URL:

```bash
# Add Hugging Face as a remote
git remote add hf https://huggingface.co/spaces/awais-dev-ai/Intern-Performance-Predictor
```

> ⚠️ Replace the URL with your actual Hugging Face Space URL. You can find it on the Space page after creation.

---

## Step 3: Push Your Code to Hugging Face

```bash
# Push your main branch to Hugging Face
git push hf main
```

This will trigger a **Docker build** on Hugging Face's servers. The build takes about **5-10 minutes** the first time (installing dependencies).

You can watch the build progress live on your Space page.

---

## Step 4: Configure Environment (Optional)

If you need to set environment variables on Hugging Face:

1. Go to your Space page
2. Click **Settings** (gear icon)
3. Scroll to **Repository Secrets**
4. Add any variables (not needed for this project — it works out of the box)

---

## Step 5: Access Your Deployed App

Once the build completes:
- Green "Running" badge appears on your Space
- Your app is live at:  
  `https://huggingface.co/spaces/awais-dev-ai/Intern-Performance-Predictor`

---

## Step 6: Verify the Deployment

1. Open the URL above
2. Fill in the form fields and click **Predict**
3. Check the `/health` endpoint: `https://huggingface.co/spaces/awais-dev-ai/Intern-Performance-Predictor/health`

Expected health response:
```json
{"status": "ok", "model_name": "XGBoost"}
```

---

## Updating the Deployed App

To update the app after making changes:

```bash
git push hf main
```

Hugging Face automatically rebuilds the Docker container on every push.

---

## Viewing Logs

To see the logs (helpful if something goes wrong):

1. Go to your Space page: `https://huggingface.co/spaces/awais-dev-ai/Intern-Performance-Predictor`
2. Click the **three dots (...)** menu → **View logs**
3. You'll see the Docker build output and any application errors

---

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Build fails | Missing `gunicorn` in requirements.txt | Ensure `gunicorn>=21.2,<23` is in requirements.txt |
| App crashes on startup | Hugging Face's internal port differs | The Dockerfile uses `EXPOSE 7860` with `ENV PORT=7860`, which Hugging Face expects (locally the app defaults to 5000 but the container overrides it) |
| Model trains every restart | `models/` not persisted | Hugging Face Spaces are ephemeral — model retrains on each restart (normal) |
| Slow first load | Docker is building dependencies | Wait 5-10 minutes. Subsequent pushes are faster due to caching |

---

## Quick Commands Summary

```bash
# First deployment
git remote add hf https://huggingface.co/spaces/awais-dev-ai/Intern-Performance-Predictor
git push hf main

# Update after changes
git push hf main

# View logs (on Hugging Face website)
# Space page → ... → View logs