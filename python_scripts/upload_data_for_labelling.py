import glob
import boto3
from loguru import logger
import sys
import json
import pandas as pd
import os
import p_tqdm
import multiprocessing
from sqlalchemy import create_engine
from moviepy.editor import VideoFileClip
print("moviepy import successful")

FPS =  30
LABELLING_S3_BUCKET = 'netradyne-labelling-production'
LABELLING_S3_PREFIX = 'dms_eec_alert_level_labelling_robust_v0.1'
VIDEO_OFFSET = 2 # offset in seconds to start the video before the event start time

def get_EEC_events(json_path: str) -> pd.DataFrame:
    """
    Extracts EEC event data from a summary JSON file.
    Args:
        json_path (str): Path to the summary JSON file.
    Returns:
        pd.DataFrame: DataFrame containing EEC event data.
    """
    rows = []
    avid_folder_name = os.path.basename(os.path.dirname(json_path))
    avid = avid_folder_name.split('__')[0]
    with open(json_path, 'r') as f:
        jsonDict = json.load(f)
    alerts = jsonDict['inference_data']['events_data']['alerts']
    for each_alert in alerts:
        if each_alert['event_code'] == "401.1.5.0.0":
            rows.append({
                'avid': avid,
                'avid_folder_name': avid_folder_name,
                'start_timestamp': each_alert['start_timestamp']/1000,
                'end_timestamp': each_alert['end_timestamp']/1000,
                'event_code': each_alert['event_code'],
                'uuid': each_alert['uuid'],
                'alert_id': each_alert['alert_id'],
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame([{
        'avid': avid,
        'avid_folder_name': avid_folder_name,
        'start_timestamp': None,
        'end_timestamp': None,
        'event_code': None,
        'uuid': None,
        'alert_id': None,
    }])

def process_video(each_df_row: dict) -> None:
    """
    Processes a single video based on metadata from a DataFrame row.
    1. Downloads video from S3.
    2. Extracts frames based on event timestamps.
    3. Saves frames locally.
    4. Uploads frames to a specified S3 location.
    Args:
        each_df_row (dict): A dictionary containing video metadata including 'avid' and 's3_bucket'.
    Returns:
        None
    """
    s3_client = boto3.client('s3')
    s3_bucket, s3_folder_prefix = each_df_row['s3_bucket'].split('/', 1)
    os.makedirs(f"temp/{each_df_row['uuid']}", exist_ok=True)
    local_video_path = os.path.join(f"temp/{each_df_row['uuid']}", "8.mp4")
    try:
        s3_client.download_file(s3_bucket, f"{s3_folder_prefix}/8.mp4", local_video_path)
        logger.debug(f"Downloaded {each_df_row['avid']} from S3 to {local_video_path}")
    except Exception as e:
        logger.error(f"Error downloading {each_df_row['avid']} from S3: {e}")
        return
    
    try:
        frames_dir = f"temp/{each_df_row['uuid']}/frames"
        os.makedirs(frames_dir, exist_ok=True)
        logger.debug(f"Extracting frames from {local_video_path} to {frames_dir}")

        video_duration = VideoFileClip(local_video_path).duration
        start_time = max(0, int(each_df_row['start_timestamp'] - VIDEO_OFFSET))  # in seconds
        end_time = min(video_duration, int(each_df_row['end_timestamp'] + VIDEO_OFFSET))  # in seconds
        duration = end_time - start_time
        os.system(f"ffmpeg -ss {start_time} -i {local_video_path} -t {duration} -vf fps={FPS} -start_number 0 {frames_dir}/frame_%04d.jpg > /dev/null 2>&1")
        logger.debug(f"Extracted frames to {frames_dir} from {start_time} to {end_time} seconds")
    except Exception as e:
        logger.error(f"Error in extracting frames for video {each_df_row['avid']}: {e}")

    # removing the entire folder to save space
    try:
        os.system(f"rm -rf temp/{each_df_row['uuid']}")
        logger.debug(f"Removed temporary files for {each_df_row['uuid']}")
    except Exception as e:
        logger.error(f"Error removing temporary files for {each_df_row['avid']}: {e}")

    return


if __name__ == "__main__":
    """
    Main function to process videos and extract frames based on EEC events.
    1. Fetches video metadata from a PostgreSQL database.
    2. Reads summary JSON files to get EEC event data.
    3. Merges event data with video metadata.
    4. Processes each video to extract and upload frames.
    """
    # Set up logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.remove()
    logger.add(sys.stderr, level=LOG_LEVEL)

    query = '''
      SELECT avid, s3_bucket
      FROM video_catalog
      WHERE dms_video_file IS NOT NULL
        AND is_external_video = FALSE
      LIMIT 1000;
    '''
    kpi_con = create_engine(f'postgresql://prithvi.ram:a40f2f11e0@analytics.cjtip3nhxyf3.us-west-1.rds.amazonaws.com:5432/kpis')
    s3_path_list = pd.read_sql_query(query, kpi_con) # reading from the kpi db
    s3_path_list = s3_path_list.astype(str) # converting everything to string
    logger.info("------------------- s3 path list -------------------")
    logger.info(f"\n{s3_path_list.head()}")

    base_dir = '/inwdata2/Prithvi/AN_25908_eec_recall_improvement/dms_submit_job_141184/'
    logger.info(f"reading summary json files from {base_dir}")
    summary_json_paths = glob.glob(f'{base_dir}*/summary.json')
    logger.info(f'Found {len(summary_json_paths)} summary.json files')

    # Run in parallel, get a list of DataFrames
    events = p_tqdm.p_map(get_EEC_events, summary_json_paths[:1000], num_cpus=multiprocessing.cpu_count(), desc='Processing EEC outputs')
    events_df = pd.concat(events, ignore_index=True)
     
    # Filter out rows with null event_code
    events_df = events_df[events_df['event_code'].notnull()]
    # converting avid, avid_folder_name event_code, uuid, alert_id to string
    for col in ['avid', 'avid_folder_name', 'event_code', 'uuid', 'alert_id']: 
        events_df[col] = events_df[col].astype(str)
    logger.info(f"Events dataframe shape: {events_df.shape}, Events DataFrame sample:\n{events_df.head()}")

    # Further filter events based on duration
    cond1 = (events_df['end_timestamp'] - events_df['start_timestamp']) >= 2
    cond2 = (events_df['end_timestamp'] - events_df['start_timestamp']) <= 2.5
    filtered_events_df = events_df[cond1 & cond2]   
    logger.info(f"filtering the df to get events which are between 2 to 2.5, and its length is {filtered_events_df.shape}")

    merged_df = pd.merge(events_df, s3_path_list, on='avid', how='inner')
    logger.info(f"Merged DataFrame shape: {merged_df.shape}")
    # process each video in parallel
    p_tqdm.p_map(process_video, 
                 merged_df.to_dict('records'), 
                 num_cpus= multiprocessing.cpu_count(), 
                 desc=' processing videos',
                 disable= LOG_LEVEL == "DEBUG")
    logger.info("Processing completed.")
