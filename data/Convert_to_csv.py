import dask.dataframe as dd

ddf = dd.read_parquet('K://Sandbox//New folder//*.parquet')
ddf.to_csv("master.csv", 
    single_file=True, 
    index=False
)