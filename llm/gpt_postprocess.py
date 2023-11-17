import os

dataset_dir = "/sbksvol/amurali/data/merged/train"

for filename in os.scandir(dataset_dir):
    if filename.is_file():
        if(os.path.splitext(filename)[1] == ".txt"):
            continue
        else:
            file_name = os.path.splitext(os.path.basename(filename))[0]
            print(filename)
