import os
import pathlib
import shutil
import requests
import yaml

from tqdm import tqdm


def download_file(url, filename, directory=''):
    with requests.get(url, stream=True, allow_redirects=True) as rhead:
        with requests.get(rhead.url, stream=True, allow_redirects=True) as r:
            dest = os.path.join(directory, filename)
            if not os.path.exists(dest):
                pathlib.Path(directory).mkdir(parents=True, exist_ok=True) 
                with open(dest, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            return dest


with open("models.yaml") as model_file:
    model_config = yaml.load(model_file, Loader=yaml.FullLoader)

dest_dir = "./source_model_files"
for model in tqdm(model_config["list"], desc="Downloading model"):
    f = download_file('/'.join([model_config["repository"], model["code"]]), model["name"], directory=dest_dir)
    model["filename"] = f

