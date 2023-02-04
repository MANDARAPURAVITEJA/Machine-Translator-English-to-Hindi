## This file for testing python commands
import os


def get_path():
    ROOT_DIR = os.getcwd()  #to get current working directory
    CONFIG_DIR = "config"
    CONFIG_FILE_NAME = "params.yaml"
    CONFIG_FILE_PATH = os.path.join(ROOT_DIR,CONFIG_DIR,CONFIG_FILE_NAME)
    print(CONFIG_FILE_PATH)
    return CONFIG_FILE_PATH

get_path()