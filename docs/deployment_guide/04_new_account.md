# 4 · New Hugging Face account? Use Streamlit

If your Hugging Face account is too new for ZeroGPU, use **Streamlit Community Cloud**. It hosts a small Python app for free and gives you a public `streamlit.app` link. You need a **GitHub account** and a **Streamlit Community Cloud account**; you do not need a Hugging Face account or a GPU.

This is a separate route. The [Hugging Face instructions](02_publish.md) still work for eligible accounts.

## Step 1 · Get the CPU starter

[Download the Streamlit starter](surface_defect_streamlit.zip) and extract it. Add your own weights so the folder contains:

```text
app.py
model.py
weights.pt
requirements.txt
```

Use the same `model.py` and `weights.pt` you prepared in [page 1](01_prepare.md). Have a `.pth` instead? Use [page 3](03_have_a_pth.md). Replace the starter's placeholder architecture with your trained architecture.

In the new `app.py`, copy your working **preprocessing, activation, threshold and mask filtering** into `preprocess()` and `predict()`. The starter assumes grayscale input and four independent sigmoid outputs, in a–d order. If your model uses softmax/background or RGB, keep its own validation-time prediction logic instead. Run on **CPU**: remove CUDA moves and ZeroGPU decorators from any code you copy. Do not copy the old `gr.Interface` or `demo.launch()` section.

Add any libraries your architecture needs to the new `requirements.txt` (for example `segmentation-models-pytorch==0.5.0`). Keep the CPU-only PyTorch requirement supplied in this starter; it saves installation space. Build pretrained encoders with `encoder_weights=None`.

## Step 2 · Test locally

Open a terminal **inside the extracted folder**:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv run --no-project streamlit run app.py
```

Open the local URL shown in the terminal. Upload a strip and press **Predict**. The result is the same 800 × 128 image with coloured masks. The uploader accepts the native aspect ratio, including proportional sizes such as 1600 × 256; prediction runs at 800 × 128.

Streamlit replaces only the interface. The end of the starter essentially does this:

```python
upload = st.file_uploader("Upload a surface strip", type=["png", "jpg", "jpeg"])
if upload is not None and st.button("Predict"):
    st.image(predict(Image.open(upload)))
```

`@st.cache_resource` above `load_model()` keeps one loaded model between uploads. Compare one output against your existing local app before publishing.

## Step 3 · Put the four files on GitHub

Create a **new public repository**, such as `surface-defect-demo`, at [github.com/new](https://github.com/new). Choose **Add a README**. Upload only the four deployment files to its root using **Add file → Upload files**, then commit. Your code and weights will be public. Never upload the competition dataset or sample competition images.

**Weights too large for the web uploader?** GitHub's browser upload limit is 25 MiB per file. Use [Git LFS](https://git-lfs.com/) for the weights; Streamlit Community Cloud supports it. After installing Git and Git LFS, clone your new repository:

```bash
git clone https://github.com/YOUR_USERNAME/surface-defect-demo.git
cd surface-defect-demo
git lfs install
git lfs track "*.pt"
```

Copy the four files into this cloned folder, then:

```bash
git add .gitattributes app.py model.py weights.pt requirements.txt
git commit -m "Add model demo"
git push origin main
```

Use LFS **before the first commit of the weights**. GitHub LFS storage/download quotas still apply; avoid uploading many checkpoint versions.

## Step 4 · Deploy

1. Open [share.streamlit.io](https://share.streamlit.io), create an account, and connect your GitHub account.
2. Click **Create app**, then **Yup, I have an app** if asked.
3. Select your repository, branch **main**, and main file **app.py**.
4. Under **Advanced settings**, select **Python 3.12**. Click **Deploy**.
5. Wait for installation, open your app, upload a strip, and press **Predict**. Share the resulting `https://…streamlit.app` link.

Pushing changes to GitHub updates the app. After replacing weights, reboot the app from **Manage app** so the cached model reloads.

## What to expect

- **CPU, not GPU:** one image at a time is suitable for a small demo. Large encoders or ensembles may be slow or exceed memory limits. Start with a single compact checkpoint.
- **Limited memory:** Streamlit currently lists roughly 690 MB–2.7 GB of memory, rather than guaranteeing the maximum. An out-of-memory error means the model/runtime needs to be smaller; changing a CUDA setting will not fix it.
- **Sleeping apps:** after 12 hours without traffic, an app can hibernate. Visit it and use the wake-up button before a demonstration.
- **Errors:** open **Manage app → logs**. Missing library? Add it to `requirements.txt`. Missing weights? Check the file is actually present, including its LFS upload. Wrong masks? Check the same preprocessing and output rules as on [page 1](01_prepare.md).

You now have a hosted image-to-overlay demo without waiting for Hugging Face account eligibility.

Official references (checked 15 November 2026): [free Community Cloud hosting](https://docs.streamlit.io/deploy/streamlit-community-cloud), [deployment steps](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [Git LFS support](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization), [resource limits and sleep](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app), and [Hugging Face ZeroGPU eligibility](https://huggingface.co/docs/hub/spaces-zerogpu). Service policies can change.

[← Back to the overview](index.html) · [Hugging Face route →](02_publish.md)
