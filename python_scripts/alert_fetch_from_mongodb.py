import pymongo
import p_tqdm
import pandas as pd

mongo_host = "mongo_dp_ro:vam1aBcp@analytics-dashboard-mongo-db.netradyne.info"
mongo_client = pymongo.MongoClient("mongodb://{}:27017/".format(mongo_host))
mongo_db = mongo_client["analytics"]

collection = mongo_db["video_requests_v2"]
query = {
        "event_code": { "$in": ["401.1.5.0.0", "401.1.5.0.20"] }
    }

cursor = collection.find(query)
print(f"total_documents - {collection.count_documents(query)}")

data = list(cursor)

def fetch_alert_id(document):
    if 'retrieved_message' in document:
        if 'videos' in document['retrieved_message']:
            matching_keys = [k for k in document['retrieved_message']['videos'].keys() if k.startswith('8_trip')]
            if len(matching_keys) > 0:
                cond1 = document['retrieved_message']['videos'][matching_keys[0]]['status'] == 1
                cond2 = document['retrieved_message']['videos'][matching_keys[0]]['timestamp'] > "2024-12-12T00:00:00.000Z"
                if cond1 and cond2:
                    ret_dict = {'alert_id': document['alert_id'], 'device_id': document['retrieved_message']['device_id']}
                    return document['alert_id']

results = p_tqdm.p_map(fetch_alert_id, data, num_cpus=48)
results = [res for res in results if res is not None]
temp_df = pd.DataFrame(results, columns=['alert_id'])
print(temp_df.tail())
print(temp_df.info())
temp_df.to_csv("labelling_eec_alert_ids.csv", index=False)