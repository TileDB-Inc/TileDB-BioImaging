import os
import urllib.parse

import boto3

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def get_path(name):
    return os.path.join(DATA_DIR, name)


def download_from_s3(uri, s3=boto3.client("s3")):
    parsed_uri = urllib.parse.urlparse(uri)
    local_path = get_path(os.path.basename(parsed_uri.path))
    if not os.path.exists(local_path):
        s3.download_file(parsed_uri.netloc, parsed_uri.path.lstrip("/"), local_path)
    return local_path
