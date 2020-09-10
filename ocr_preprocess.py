import os
import shlex
import subprocess

def preprocessing(file_name):
    os.system(f"chmod -wx ./imgtxtenh/{file_name}")
    os.system(f"./imgtxtenh/imgtxtenh ./imgtxtenh/{file_name} -p ./imgtxtenh/pre_{file_name}")
