import os
import zipfile
import gdown

google_file_id = '13E8dQMOpqj522gHR1mfJps9JGkIJzITD'
zip_file_name = 'data.zip'

if "data" not in os.listdir():
    gdown.download(f'https://drive.google.com/uc?id={google_file_id}', zip_file_name, quiet=False)

    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall()

    os.remove(zip_file_name)
