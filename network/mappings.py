import numpy as np

def map_scene_array_to_dict(array):
    return dict(zip(['frame_index_interval', 'host', 'start_time', 'end_time'], array))
def map_frame_array_to_dict(array):
    return dict(zip(['timestamp', 'agent_index_interval', 'ego_blank', 'ego_translation','ego_rotation'], array))
def map_agent_array_to_dict(array):
    return dict(zip(['centroid', 'extent', 'yaw', 'velocity', 'track_id'], array))
def map_ret_to_dict(array):
    return dict(zip(['coordinates','velocity','yaw','tags'],array))
def getWithout(arr,n):
    return arr[arr!=n]