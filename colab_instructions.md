# Running Agricultural Price Prediction on Google Colab TPU

This guide explains how to run your project on Google Colab to leverage TPU (Tensor Processing Unit) acceleration for faster model training.

## 1. Prepare Google Colab Environment

1.  Open [Google Colab](https://colab.research.google.com/).
2.  Create a **New Notebook**.
3.  Go to **Runtime** > **Change runtime type**.
4.  Select **Python 3** as the runtime type and **TPU** as the hardware accelerator.
5.  Click **Save**.

## 2. Upload Your Project

You have a few options to get your code into Colab. The easiest is often to zip your project folder and upload it.

### Option A: Zip and Upload
1.  Zip your entire `Miniproject` folder on your local machine.
2.  In the Colab notebook, verify you are connected (RAM/Disk bar in top right).
3.  Click the **Files** folder icon on the left sidebar.
4.  Click the **Upload** icon and select your `Miniproject.zip` file.
5.  Unzip the file:
    ```python
    !unzip Miniproject.zip
    ```

### Option B: Clone from GitHub (if pushed)
1.  If your code is on GitHub:
    ```python
    !git clone <your-repository-url>
    ```

## 3. Setup Dependencies

Install the required libraries. Run this in a code cell:

```python
%cd Miniproject
!pip install -r requirements.txt
```

> **Note:** If you don't have a `requirements.txt`, you can install packages manually:
> ```python
> !pip install pandas numpy scikit-learn tensorflow statsmodels
> ```

## 4. Run the Models

Now you can run your training scripts. The code has been updated to automatically detect the TPU.

### Run Deep Learning Model
```python
!python src/models/deep_learning.py
```

### Run Hybrid Model
```python
!python src/models/hybrid.py
```

## 5. Verify TPU Usage

When the script runs, check the output logs. You should see a message confirming TPU detection:
`INFO:__main__:Running on TPU: grpc://...`

If you see `TPU not found, using default strategy`, double-check that your Runtime type is set to TPU.
