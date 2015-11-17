import pandas as pd
import sys
import os

def simplify(infile,outfile):
    print(infile)
    df = pd.DataFrame.from_csv(infile, index_col=0, sep='\t')
    print(df)
    df = df.reindex_axis([c for c in df.columns[1:] if not c.startswith('Unnamed')], 1)
    renamed_columns = {c: c[:c.find('_')] for c in df.columns}
    df = df.rename(columns=renamed_columns)
    df = df.drop(labels=[x for x in df.index if not type(x) == str or not x[0].isupper()])
    df = df.groupby(axis=1,level=0).first()
    df = df.groupby(axis=0,level=0).first()
    df.index.name = "Genes"
    df.columns.name = "Cell_Lines"
    df.to_csv(outfile,sep="\t")

if __name__ == "__main__":
    simplify(os.path.dirname(__file__) + "/../../" + sys.argv[1],os.path.dirname(__file__) + "/../../"  + sys.argv[2])