import glob
import os
import p_tqdm
import pandas as pd
from loguru import logger
import subprocess
import boto3
import sys
import json
from io import StringIO

BUCKET_NAME = "netradyne-labelling-production"
S3_FOLDER_PATH = "dms_eec_alert_level_labelling_AN25908_v0.3"

def extract_label_info(uuid_folder_name: str) -> dict:
    """
    Extracts label information from the labels.txt file in the specified S3 folder.
    1. Connects to S3 and lists objects in the annotations_attribute folder.
    2. Reads the labels.txt file into a pandas DataFrame.
    3. Cleans the data by removing quotes and dropping null labels.
    4. Constructs a dictionary with category as keys and start_frame, end_frame, and label_value as values.
    Args:
        uuid_folder_name (str): The UUID folder name in the S3 path.
    Returns:
        dict: A dictionary with category as keys and a nested dictionary containing
                start_frame, end_frame, and label_value as values.
    """
    label_info = None
    try:
        s3_client = boto3.client("s3")
        response = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=f"{S3_FOLDER_PATH}/{uuid_folder_name}/annotations_attribute/"
        )
        if 'Contents' not in response:
            logger.warning(f"No annotations_attribute folder found for {uuid_folder_name}")
            return label_info
    except Exception as e:
        logger.error(f"Error accessing S3 for folder {uuid_folder_name}: {e}")
        return label_info
    
    try:
        labels_file_key = f"{S3_FOLDER_PATH}/{uuid_folder_name}/annotations_attribute/latest/labels.txt"
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=labels_file_key)
        labels_content = obj['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(labels_content),
                         sep=" ",
                         header=None,
                         names=['checkbox_id', 'frame_idx', 'category', 'label'])
        logger.debug(f"shape of labels.txt for {uuid_folder_name}: {df.shape}")

        # remove quotes and clean data
        df['category'] = df['category'].str.replace('"', '', regex=False)
        df['label'] = df['label'].astype(str).str.replace('"', '', regex=False)
        df = df[df['label'].notnull() & (df['label'] != 'nan')]
        logger.debug(f"shape of labels.txt after dropping nulls for {uuid_folder_name}: {df.shape}")

        # Building dictionary
        label_info = (
            df.groupby('category')
              .agg(start_frame=('frame_idx', 'min'),
                   end_frame=('frame_idx', 'max'),
                   label_value=('label', 'first'))   # take first non-null label
              .to_dict(orient='index')
        )
        
    except Exception as e:
        logger.error(f"Error accessing labels.txt for folder {uuid_folder_name}: {e}")
        return label_info
    return label_info


if __name__ == "__main__":
    """
    Main execution block.
    Sets up logging, retrieves folder list from S3, and extracts label information in parallel.
    1. Configures logging based on environment variable.
    2. Lists all folders in the specified S3 path.
    3. Uses p_tqdm to parallelize label extraction across multiple CPU cores.
    4. Combines extracted label information into a single dictionary.
    """

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.remove()
    logger.add(sys.stderr, level=LOG_LEVEL)

    # get length of all folders in the s3 path
    s3_path = f"s3://{BUCKET_NAME}/{S3_FOLDER_PATH}/"
    result = subprocess.run(["aws", "s3", "ls", s3_path],
                            capture_output=True,
                            text=True)
    folders = result.stdout.strip().splitlines()
    logger.info(f"Found {len(folders)} folders in S3 path: {s3_path}")
    folders = [f.replace("PRE", "").strip().replace("/", "") for f in folders]
    logger.info(f"sample folders: {folders[:5]}")
    # extract label info for all folders in parallel
    all_label_info = p_tqdm.p_map(extract_label_info,
                                  folders[:500],  # limiting to first 500 for testing
                                  num_cpus=48,
                                  desc="Extracting label info",
                                  disable=False)  
    # combine into a single dict
    combined_label_info = dict(zip(folders[:500], all_label_info))
    logger.info(f"type of combined_label_info: {type(combined_label_info)}")
    with open("/inwdata2/Prithvi/GIT/work/AN25908/eec_annotations.json", "w") as f:
        json.dump(combined_label_info, f, indent=4)
    logger.info("Saved combined_label_info to eec_annotations.json")


    
