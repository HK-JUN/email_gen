import gdown
import pandas as pd

file_id = "1URNq8vGbhDNBhu_UfD9HrEK8bkgWcqpM"
destination = "politeness.csv"
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url,destination,quiet=False)
df = pd.read_csv(destination)
print(df.head())