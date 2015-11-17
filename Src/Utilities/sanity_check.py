import pandas as pd
df = pd.DataFrame.from_csv("Data/CCLE_Data/CCLE_Expression_2012-09-29.res", index_col=0, sep='\t')
print("RPL6" in df.index)
print("RPL10A" in df.index)
print("RPS8" in df.index)
