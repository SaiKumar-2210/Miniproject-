# Training Miniproject Models on Google Colab

If your PC is struggling to train the models (especially the Deep Learning LSTM and Hybrid models), you can use Google Colab's free GPUs/TPUs to train them and then copy the trained models back to your PC.

Here is the exact step-by-step process.

---

## 1. Prepare Google Colab Environment

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a **New Notebook**.
3. Go to **Runtime** > **Change runtime type**.
4. Select **Python 3** as the runtime type and **T4 GPU** or **TPU** as the hardware accelerator.
5. Click **Save**.

---

## 2. Upload Your Project and Dataset

You need to get your code and the processed data into Colab.

### Option A: Zip and Upload (Recommended)
1. On your PC, zip the `src`, `data`, and `config` folders into a file called `Miniproject.zip`. 
   *(Make sure `data/processed/features_data.csv` is included inside the zip).*
2. In the Colab notebook, click the **Folder icon** on the left sidebar.
3. Drag and drop `Miniproject.zip` into the file explorer area.
4. Run this in a code cell to unzip:
   ```bash
   !unzip Miniproject.zip
   ```
5. Ensure the extracted folders (`src`, `data`, `config`) are in the main `/content/` directory.

### Option B: GitHub + Google Drive (For Large Datasets)
1. Run this to clone your code (if it's on GitHub):
   ```bash
   !git clone <your-repository-url>
   %cd Miniproject
   ```
2. Run this cell to mount your Google Drive (where you can upload `features_data.csv`):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Copy the dataset from Drive to the project folder:
   ```bash
   !mkdir -p data/processed
   !cp /content/drive/MyDrive/features_data.csv data/processed/
   ```

---

## 3. Install Requirements

Run this cell to install the necessary libraries:

```bash
!pip install pandas numpy scikit-learn tensorflow statsmodels arch pyyaml
```

*(Note: Colab already has TensorFlow, Pandas, and Numpy installed, so this just ensures you have statsmodels and arch for the ARIMA/GARCH components).*

---

## 4. Run the Training Scripts

Your code is already set up to automatically use the Colab GPU/TPU if it's available.

### Train the Deep Learning (LSTM) Model:
```bash
!python src/models/deep_learning.py
```

### Train the Hybrid Model:
```bash
!python src/models/hybrid.py
```

### Run the Full Evaluation Pipeline:
```bash
!python src/models/evaluate.py
```

---

## 5. Download the Trained Models Back to Your PC

Once training finishes, the models and scalers are saved in the `models/lstm/` and `models/arima/` folders inside Colab. 

You need to download them back to your PC so your frontend/API can use them.

1. **Zip the models folder** in Colab:
   ```bash
   !zip -r trained_models.zip models/
   ```

2. **Download the zip file** directly via code:
   ```python
   from google.colab import files
   files.download('trained_models.zip')
   ```
   *(Or just right-click `trained_models.zip` in the Colab file explorer and click Download).*

3. **Extract on your PC**:
   Extract `trained_models.zip` and move the contents into your local `c:\Users\SaiKumar\Documents\BTECH-III\Miniproject\models\` directory, overwriting the old files.

4. **Copy the Evaluation CSV**:
   Don't forget to also download and replace your `models/evaluation_results.csv` to update your dashboard metrics!
