from loguru import logger
import p_tqdm
import subprocess
import os
import boto3
import pandas as pd

BUCKET_NAME = "netradyne-labelling-production"
S3_FOLDER_PATH = "dms_eec_alert_level_labelling_AN25908_v0.3"
# s3://netradyne-labelling-production/dms_eec_alert_level_labelling_AN25908_v0.3/0003ecdf-d455-4113-a8a1-6d6f0a0cec36/vframes/0/0/
def image_folder_downloader(uuid_folder_name: str)-> None:
    try:
        img_folder_path = f"{S3_FOLDER_PATH}/{uuid_folder_name}/vframes/0/0/"
        local_folder_path = f"temp/{uuid_folder_name}/"
        os.makedirs(local_folder_path, exist_ok=True)
        os.system(f"aws s3 cp s3://{BUCKET_NAME}/{img_folder_path} {local_folder_path} --recursive --quiet")
        logger.info(f"Downloaded images for folder {uuid_folder_name} to {local_folder_path}")
    except Exception as e:
        logger.error(f"Error downloading images for folder {uuid_folder_name}: {e}")
        return None
        

if __name__ == "__main__":
    os.system("rm -rf temp")
    uuids = pd.read_csv("/inwdata2/Prithvi/GIT/work/AN25908/eec_fn_high_confidence_uuids.csv")['uuid'].tolist()
    p_tqdm.p_map(image_folder_downloader, uuids, num_cpus=48, desc="Downloading image folders", disable=True)