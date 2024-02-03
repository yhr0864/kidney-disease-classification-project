#create the empty template folders and files for the future projects

import os
from pathlib import Path
import logging

#logging string
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = 'kidney disease classification'

list_of_files = [
    ".github/workflow/.gitkeep", #https://www.freecodecamp.org/news/what-is-gitkeep/         
                                 #even though the dir /.../workflow is empty, it can still be tracked by git.
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml", 
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html" #use flask framework                        
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "": #create dirs/folders
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"creating directory {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0): #create files
        with open(filepath, 'w') as f:
            pass
            logging.info(f"creating empty file {filepath}")

    else:
        logging.info(f"{filename} already exists")
