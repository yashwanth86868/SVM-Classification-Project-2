import shutil
import os

os.makedirs("data", exist_ok=True)

shutil.copy(
    r"C:\Users\ybali\Downloads\logit classification.csv",
    "data/logit_classification.csv"
)

print("CSV file copied successfully!")