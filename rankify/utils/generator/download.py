import os
import requests
import tarfile
from tqdm import tqdm

class ModelDownloader:
    """
    Utility class for downloading and extracting model files.
    """

    @staticmethod
    def download_and_extract(url, output_dir):
        """
        Downloads and extracts a model from a given URL.
        """
        os.makedirs(output_dir, exist_ok=True)
        tar_path = os.path.join(output_dir, "model.tar.gz")

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        with open(tar_path, "wb") as file, tqdm(
            desc="Downloading Model", total=total_size, unit="B", unit_scale=True
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))

        with tarfile.open(tar_path, "r:gz") as tar:
            subdir = tar.getnames()[0]  # Get the top-level directory inside the tarball
            tar.extractall(output_dir)
            
        # Move contents up if they were extracted into a subdirectory
        extracted_path = os.path.join(output_dir, subdir)
        if os.path.exists(extracted_path) and os.path.isdir(extracted_path):
            for file in os.listdir(extracted_path):
                os.rename(os.path.join(extracted_path, file), os.path.join(output_dir, file))
            os.rmdir(extracted_path)  # Remove the now-empty subdirectory
