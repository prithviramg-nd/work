import glob
import boto3
from loguru import logger
import sys
import json
import pandas as pd
import numpy as np
import os
import p_tqdm
import multiprocessing
from sqlalchemy import create_engine
from moviepy.editor import VideoFileClip
from PIL import ImageDraw, Image

FPS =  10
LABELLING_S3_BUCKET = 'netradyne-labelling-production'
LABELLING_S3_PREFIX = 'dms_eec_alert_level_labelling_AN25908_v0.5'
VIDEO_OFFSET = 2000 # offset in milliseconds to start the video before the event start time
IMAGE_INDEX_OFFSET = 6 # number of frames to include before the event start frame

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
    eec_event_obs = jsonDict['inference_data']['dms']['dms_drowsy']['extended_ec']['event_info']
    eec_event_index = 0
    for each_alert in alerts:
        if each_alert['event_code'] == "401.1.5.0.0":
            rows.append({
                'avid': avid,
                'avid_folder_name': avid_folder_name,
                'start_timestamp': (eec_event_obs[eec_event_index]['st_fidx'] - IMAGE_INDEX_OFFSET)*100,  # converting to milliseconds
                'end_timestamp': eec_event_obs[eec_event_index]['et_fidx']*100,  # converting to milliseconds
                'event_code': each_alert['event_code'],
                'uuid': each_alert['uuid'],
                'alert_id': each_alert['alert_id'],
            })
            eec_event_index += 1
    return pd.DataFrame(rows) if rows else pd.DataFrame([{
        'avid': avid,
        'avid_folder_name': avid_folder_name, 
        'start_timestamp': None,
        'end_timestamp': None,
        'event_code': None,
        'uuid': None,
        'alert_id': None,
    }])

def calculate_frame_numbers(video_start: int,
                            video_end: int,
                            event_start: int,
                            event_end: int,
                            each_df_row: dict,
                            fps: int = FPS) -> tuple[int, int, int, int]:
    """
    Calculates frame numbers to extract from a video based on event timestamps.
    Args:
        video_start (int): Start time of the video in milliseconds.
        video_end (int): End time of the video in milliseconds.
        event_start (int): Start time of the event in milliseconds.
        event_end (int): End time of the event in milliseconds.
        each_df_row (dict): Metadata for the video/event.
        fps (int): Frames per second of the video.
    Returns:
        tuple[int, int, int, int]: A tuple containing the start frame, end frame, event start frame, and event end frame.
    """
    event_start = max(video_start, event_start)
    event_end = min(video_end, event_end)
    event_duration = event_end - event_start
    assert event_duration > 0, f"Event duration must be positive, and corresponding df row is {each_df_row}"
    event_start_frame_idx = int((event_start - video_start) * fps / 1000)
    event_end_frame_idx = int((event_end - video_start) * fps / 1000)
    return event_start_frame_idx, event_end_frame_idx

def annotate_frames(frame_files: list[str], 
                    event_start_frame: int, 
                    event_end_frame: int, 
                    each_df_row: dict) -> None:
    """
    Annotates frames when the event occurs.
    Args:
        frame_files (list[str]): List of frame file paths.
        event_start_frame (int): Frame index where the event starts.
        event_end_frame (int): Frame index where the event ends.
        each_df_row (dict): Metadata for the video/event.
    Returns:
        None
    """
    for idx, frame_file in enumerate(frame_files):
        try:
            with Image.open(frame_file) as img:
                draw = ImageDraw.Draw(img)
                countdown = idx
                while (countdown >= 0): # annotating the frame with event start and end frames
                    color = (255, 0, 0) if event_start_frame <= countdown <= event_end_frame else (255, 255, 255) # Red for event frames, White otherwise
                    y_center = 120 if event_start_frame <= countdown <= event_end_frame else 200
                    x_center = (img.width // len(frame_files)) * countdown + 10
                    draw.circle((x_center, y_center), 5, fill=color)
                    if countdown != idx:
                        prev_x_center = (img.width // len(frame_files)) * (countdown + 1) + 10
                        prev_y_center = 120 if event_start_frame <= (countdown + 1) <= event_end_frame else 200
                        draw.line((prev_x_center, prev_y_center, x_center, y_center), fill=color, width=2)
                    countdown -= 1
                img.save(frame_file)
        except Exception as e:
            logger.error(f"Error annotating frame {frame_file}: {e} and corresponding df row is {each_df_row}")
    return 

def process_video(each_df_row: dict,
                  lock= None) -> None:
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
    # Download video from S3
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
    
    # Extract frames using ffmpeg
    try:
        frames_dir = f"temp/{each_df_row['uuid']}/frames"
        os.makedirs(frames_dir, exist_ok=True)
        logger.debug(f"Extracting frames from {local_video_path} to {frames_dir}")

        video_duration = VideoFileClip(local_video_path).duration * 1000  # in milliseconds
        start_time = max(0, int(each_df_row['start_timestamp'] - VIDEO_OFFSET))  # in milliseconds
        end_time = min(video_duration, int(each_df_row['end_timestamp'] + VIDEO_OFFSET))  # in milliseconds
        duration = end_time - start_time
        os.system(f"ffmpeg -ss {start_time/1000} -i {local_video_path} -t {duration/1000} -vf fps={FPS} -start_number 0 {frames_dir}/%d.jpg > /dev/null 2>&1")
        logger.debug(f"Extracted frames to {frames_dir} from {start_time} to {end_time} seconds")
    except Exception as e:
        logger.error(f"Error in extracting frames for video {each_df_row['avid']}: {e}")

    # annotate frames
    try:
        frame_files = glob.glob(f"{frames_dir}/*.jpg") # frame files containing full path
        event_start_frame, event_end_frame = calculate_frame_numbers(video_start=start_time,
                                                                     video_end=end_time,
                                                                     event_start=each_df_row['start_timestamp'],
                                                                     event_end=each_df_row['end_timestamp'], 
                                                                     each_df_row=each_df_row)
        annotate_frames(frame_files, event_start_frame, event_end_frame, each_df_row)
        logger.debug(f"frames are annotated present in the folder: {frames_dir}") 
    except Exception as e:
        logger.error(f"Error annotating frames for video {each_df_row['avid']}: {e}")

    # Upload frames to S3
    temp_uuid = each_df_row['uuid']
    formatted_uuid = f"{temp_uuid[:8]}-{temp_uuid[8:12]}-{temp_uuid[12:16]}-{temp_uuid[16:20]}-{temp_uuid[20:]}"
    try:
        for frame_file in frame_files:
            s3_folder_path = f"s3://{LABELLING_S3_BUCKET}/{LABELLING_S3_PREFIX}/{formatted_uuid}/vframes/0/0/"
            os.system(f"aws s3 cp {frame_file} {s3_folder_path} --quiet")
        logger.debug(f"Uploaded frames for {each_df_row['avid']} to {s3_folder_path}")
    except Exception as e:
        logger.error(f"Error uploading frames for video {each_df_row['avid']} to S3: {e}")

    # removing the entire folder to save space
    try:
        os.system(f"rm -rf temp/{each_df_row['uuid']}")
        logger.debug(f"Removed temporary files for {each_df_row['uuid']}")
    except Exception as e:
        logger.error(f"Error removing temporary files for {each_df_row['avid']}: {e}")
    if lock is not None:
        with lock:
            with open("processed_uuids.txt", "a") as f:
                f.write(f"{each_df_row['uuid']}\n")
            os.system("cat processed_uuids.txt | wc -l")
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
    LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
    logger.remove()
    logger.add(sys.stderr, level=LOG_LEVEL)
    # creating a new log file
    logger.add("upload_data_for_labelling.log", rotation="0", level=LOG_LEVEL, mode= "w")

    # add your AWS credentials here if not already configured in your environment
    # os.environ['AWS_ACCESS_KEY_ID'] = ""
    # os.environ['AWS_SECRET_ACCESS_KEY'] = ""
    # os.environ['AWS_SESSION_TOKEN'] = ""

    # Check AWS credentials
    if 'AWS_ACCESS_KEY_ID' not in os.environ:
        logger.error("ERROR: AWS_ACCESS_KEY_ID is not set. Exiting.")
        sys.exit(1)
    # removing old temp folder if exists
    if os.path.exists("temp"):
        os.system("rm -rf temp")

    # # fetch video s3 paths from the database
    # s3_path_list = pd.read_csv('/inwdata2/Prithvi/GIT/work/AN25908/eec_69k_labelling_with_avid_s3Path.csv') # reading from a csv file
    # s3_path_list = s3_path_list.astype(str) # converting everything to string
    # logger.info("------------------- s3 path list -------------------")
    # logger.info(f"\n{s3_path_list.head()}")

    # # read summary json files to get EEC events
    # base_dir = '/inwdata2/Prithvi/AN_25908_eec_recall_improvement/dms_submit_job_141184/'
    # logger.info(f"reading summary json files from {base_dir}")
    # summary_json_paths = glob.glob(f'{base_dir}*/summary.json')
    # logger.info(f'Found {len(summary_json_paths)} summary.json files')

    # # Run in parallel, get a list of DataFrames
    # # summary_json_paths = np.random.choice(summary_json_paths, size=2000, replace=False) # taking a random subset
    # events = p_tqdm.p_map(get_EEC_events, summary_json_paths, num_cpus=multiprocessing.cpu_count(), desc='Processing EEC outputs')
    # events_df = pd.concat(events, ignore_index=True)

    # # Filter out rows with null event_code
    # events_df = events_df[events_df['event_code'].notnull()]
    # # converting avid, avid_folder_name event_code, uuid, alert_id to string
    # for col in ['avid', 'avid_folder_name', 'event_code', 'uuid', 'alert_id']: 
    #     events_df[col] = events_df[col].astype(str)
    # logger.info(f"Events dataframe shape: {events_df.shape}, Events DataFrame sample:\n{events_df.head()}")

    # merged_df = pd.merge(events_df, s3_path_list, on='avid', how='inner')
    # merged_df.to_csv('/inwdata2/Prithvi/GIT/work/AN25908/avid_uuid_s3_path.csv', index=False) # saving the merged df for future reference
    # logger.info(f"Merged DataFrame sample:\n{merged_df.head()}")
    # logger.info(f"Merged DataFrame shape: {merged_df.shape}")

    merged_df = pd.read_csv('/inwdata2/Prithvi/GIT/work/AN25908/avid_uuid_s3_path.csv') # reading the merged df from a csv file
    merged_df = merged_df.sample(frac=1).reset_index(drop=True) # random shuffling the dataframe
    logger.info(f"Merged DataFrame sample:\n{merged_df.head()}")
    logger.info(f"Merged DataFrame shape: {merged_df.shape}")
    cond1 = (merged_df['end_timestamp'] - merged_df['start_timestamp']) >= 1700 # at least 1700 milliseconds
    cond2 = (merged_df['end_timestamp'] - merged_df['start_timestamp']) <= 2500 # at most 2500 milliseconds
    window_filtered_events_df = merged_df[cond1 & cond2]   
    logger.info(f"filtering the df to get events which are between 1.7 to 2.5, and its length is {window_filtered_events_df.shape}")

    # read already processed uuids
    if os.path.exists("processed_uuids.txt"):
        with open("processed_uuids.txt", "r") as f:
            processed_uuids = f.read().splitlines()
        filtered_events_df = window_filtered_events_df[~window_filtered_events_df['uuid'].isin(processed_uuids)]
    else:
        filtered_events_df = window_filtered_events_df
    logger.info(f"After removing already processed uuids, {filtered_events_df.shape} rows remain to be processed.")

    lock = multiprocessing.Lock() # to prevent multiple processes from writing to the log file simultaneously

    # process each video in parallel
    p_tqdm.p_map(lambda row: process_video(row, lock=lock),
                 filtered_events_df.to_dict('records'), 
                 num_cpus= 12, 
                 desc=' processing videos',
                 disable= LOG_LEVEL == "DEBUG")
    # process_video(filtered_events_df.to_dict('records')[0]) # for testing purpose only processing some video
    logger.info("Processing completed.")
