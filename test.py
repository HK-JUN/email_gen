import pandas as pd

dt = pd.read_csv("/home/user3/workplace/dataset/test/combined_test.csv")
seed =10
for i in range(5):
    print(f"politeness: {dt['p_tag'][i+seed]} / email content:{dt['txt'][i+seed]}")