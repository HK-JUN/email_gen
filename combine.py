import pandas as pd
import glob

path = "/home/user3/workplace/dataset/"
all_train_files = glob.glob(path+"train/train_part_*.csv")
all_test_files = glob.glob(path+"test/test_t_part_*.csv")

li = []
i=0
for filename in all_train_files:
    df = pd.read_csv(filename)
    li.append(df)
    print(f"train_chunk_{i} is appended")

# 모든 청크를 하나의 데이터프레임으로 결합
combined_df = pd.concat(li, axis=0, ignore_index=True)
combined_df.to_csv('/home/user3/workplace/dataset/train/combined_train.csv', index=False)
print("train file finished")
li = []
for filename in all_test_files:
    df = pd.read_csv(filename)
    li.append(df)
    print(f"test_chunk_{i} is appended")

combined_df = pd.concat(li, axis=0, ignore_index=True)
combined_df.to_csv('/home/user3/workplace/dataset/test/combined_test.csv', index=False)

print("All chunks combined and saved successfully!")