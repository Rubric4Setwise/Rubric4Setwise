import os
import requests
from rankify.utils.pre_defined_datasets import HF_PRE_DEFIND_DATASET
from tqdm import tqdm
import os
from pathlib import Path

def get_cache_dir():
    """Get cache directory, with fallback if not set"""
    if 'RERANKING_CACHE_DIR' not in os.environ:
        DEFAULT_CACHE_DIR = str(Path.home() / ".cache" / "rankify")
        os.environ['RERANKING_CACHE_DIR'] = DEFAULT_CACHE_DIR
        Path(DEFAULT_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    return os.environ['RERANKING_CACHE_DIR']


class DownloadManger:
    @staticmethod
    def download(retriever: str, dataset: str, force_download: bool = True) ->str:
        cache_dir = get_cache_dir()


        if retriever not in HF_PRE_DEFIND_DATASET:
            raise FileNotFoundError(f"Retriever {retriever} Not Supported yet. Please choose another retriever.\nCheck Dataset.available_dataset()")
        if dataset not in HF_PRE_DEFIND_DATASET[retriever]:
            raise FileNotFoundError(f"Dataset {dataset} Not Supported yet. Please choose another dataset.\nCheck Dataset.available_dataset()")

        filename = HF_PRE_DEFIND_DATASET[retriever][dataset]['filename']
        if '-' in dataset:
            dataset_name, dataset_split = dataset.split('-', 1)
        else:
            dataset_name = dataset
        urls = HF_PRE_DEFIND_DATASET[retriever][dataset]['url']
        path = os.path.join(cache_dir, 'dataset', retriever, dataset_name)
        file_path = os.path.join(path, filename)

        # If force_download is False and file already exists, skip downloading
        if not force_download and os.path.exists(file_path):
            print(f"File {file_path} already exists. Skipping download.")
            return file_path

        os.makedirs(path, exist_ok=True)

        for url in urls:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                with open(file_path, 'wb') as file, tqdm(
                    desc=f"Downloading {retriever} {dataset_name} {filename}",
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    total=total_size,
                ) as bar:
                    # Update progress bar while streaming chunks
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:  # Filter out keep-alive chunks
                            file.write(chunk)
                            bar.update(len(chunk))
                
                return file_path
            else:
                raise Exception(f'Failed to download the file from {url}')


