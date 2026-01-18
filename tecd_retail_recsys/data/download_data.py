import os
from tecd_retail_recsys.data.utils import download_dataset

def download_tecd_data(domains=['retail'], day_begin=1082, day_end=1308, max_workers=10):
    try:
        HF_TOKEN = os.getenv('HF_TOKEN')
    except Exception as e:
        print(e)
        raise ValueError("HF_TOKEN is not set")

    download_dataset(
        token=HF_TOKEN,
        dataset_path="dataset/small",
        local_dir="t_ecd_small_partial",
        domains=domains,
        day_begin=day_begin,
        day_end=day_end,
        max_workers=max_workers
    )
