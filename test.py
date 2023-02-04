## This file for testing python commands
import os
'''

def get_path():
    ROOT_DIR = os.getcwd()  #to get current working directory
    CONFIG_DIR = "config"
    CONFIG_FILE_NAME = "params.yaml"
    CONFIG_FILE_PATH = os.path.join(ROOT_DIR,CONFIG_DIR,CONFIG_FILE_NAME)
    print(CONFIG_FILE_PATH)
    return CONFIG_FILE_PATH

get_path()'''

from pathlib import Path
import os

# prefix components:
space =  '    '
branch = '│   '
# pointers:
tee =    '├── '
last =   '└── '


def tree(dir_path: Path, prefix: str=''):
    """A recursive generator, given a directory Path object
    will yield a visual tree structure line by line
    with each line prefixed by the same characters
    """    
    contents = list(dir_path.iterdir())
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir(): # extend the prefix and recurse:
            extension = branch if pointer == tee else space 
            # i.e. space because last, └── , above so no more |
            yield from tree(path, prefix=prefix+extension)


for line in tree(Path().absolute()/'templates'):
    print(line)
##Path.Home() gives c:/Users/ravit/<add required path here>
#print(Path().absolute())