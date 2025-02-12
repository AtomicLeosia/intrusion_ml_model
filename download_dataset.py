import os
import urllib.request
import zipfile

def retrieve():
    dataset_uri = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip"
    source_dir = "MachineLearningCVE"
    target_dir = "dataset"

    zip_filename = os.path.basename(dataset_uri)
    print("Downloading dataset in progress...\n")
    urllib.request.urlretrieve(dataset_uri, zip_filename)

    with zipfile.ZipFile(zip_filename, 'r') as zip:
        zip.extractall()

    if os.path.isdir(source_dir):
        os.rename(source_dir, target_dir)
        
