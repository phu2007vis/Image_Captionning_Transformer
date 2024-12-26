import os
import requests
import tempfile
from tqdm import tqdm


def download_weights(uri, cached=None, md5=None, quiet=False):
    if uri.startswith("http"):
        return download(url=uri, quiet=quiet)
    return uri


def download(url,tmp_dir = None,name = None, quiet=False):
    if tmp_dir is None:
        tmp_dir = "/work/21013187/tmp"
    if name is  None:
        filename = url.split("/")[-1]
        full_path = os.path.join(tmp_dir, filename)
    else:
        full_path = os.path.join(tmp_dir, name)
    # import pdb;pdb.set_trace()
    if os.path.exists(full_path):
        print("Model weight {} exsits. Ignore download!".format(full_path))
        return full_path

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(full_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)
    return full_path
