import yaml
import pandas as pd
import argparse
import os
import ast


def read_params(config_path):
    with open(config_path) as f:
        params = yaml.safe_load(f)
        return params


def get_data(config_path):
    config = read_params(config_path)
    path=os.getcwd()
    path=path[:-8]
    data_path = config['data_source']['master_file']
    config_path=path+data_path
    #df = pd.read_csv(path+data_path, sep=',', encoding='utf8', index_col=0)
    df=pd.read_csv(path+data_path,header=None)
    df=df.drop(df.index[0])
    df1=[]
    for i in df[0]:
        dictionary = ast.literal_eval(i)
        df1.append([dictionary['en'],dictionary['hi']])
    df1=pd.DataFrame(df1)
    print(df1.head())
    return df1

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    ROOT_DIR = os.getcwd()  #to get current working directory
    CONFIG_DIR = "config"
    CONFIG_FILE_NAME = "params.yaml"
    CONFIG_FILE_PATH = os.path.join(ROOT_DIR,CONFIG_DIR,CONFIG_FILE_NAME)
    args.add_argument("--config",type=str, default=CONFIG_FILE_PATH)
    args.add_argument("--data_path", type=str, default="./data")
    parsed_args = args.parse_args()
    print(parsed_args.config)
    data = get_data(parsed_args.config)
    print(data)