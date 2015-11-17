import pandas as pd
import Src.DataFormatter as data

def check_drug_8_response_rate(ic50_file,drug):
    df = pd.DataFrame.from_csv(ic50_file,index_col=2,sep=",")[['Compound','IC50..uM.']]
    df = df[(df['Compound'] == drug)]
    return float(len(df[df["IC50..uM."] == "8"].index)) / float(len(df.index))

if __name__ == '__main__':
    drugs = ["Paclitaxel","Erlotinib","Nilotinib","AZD6244"]
    for drug in drugs:
         print("%s: %s" % (drug, str(data.check_drug_8_response_rate("Data/IC_50_Data/CL_Sensitivity_Multiple_Drugs.csv",drug))))