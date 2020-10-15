import numpy as np

FRAMES_PER_SECOND = 10

# find the angle between two vectors
angle_between = lambda u, v: np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

# make it so we can request only a certain track id for angular velocity
# this needs to be tweaked. we dont need every angular velocity for every car
class AngularVelocityComputer:
    def __init__(self, scene, frames, agents, window=10, nan_replacement=0):
        self._fii = scene['frame_index_interval']
        self._frames = frames
        self._agents = agents
        self._window = window
        self._nan_replacement = nan_replacement
        
        # map_ftiti is map frame, track id to index
        # maybe don't name it so strangly
        self._max_track_id, self._map_ftiti = self._extract_track_ids()

        import time

        start = time.time()
                
        self._moving_angular_velocities = [self._compute_moving_angular_velocity(tid)
                                           for tid in range(*[1, self._max_track_id + 1])]

        end = time.time()

        print(end - start)
    
    def __getitem__(self, i):
        return self._moving_angular_velocities[i]
    
    def change_window_and_recompute(self, window=10):
        self._window = window 
        self._moving_angular_velocity = [self._compute_moving_angular_velocity(tid)
                                         for tid in range(*[1, self._max_track_id + 1])]
    
    def _extract_track_ids(self):
        max_track_id = 0
        map_frame_track_id_to_index = dict()
        
        for frame_i in range(*self._fii):
            if frame_i not in map_frame_track_id_to_index:
                map_frame_track_id_to_index[frame_i] = dict()
            
            frame = self._frames[frame_i]
            aii = frame['agent_index_interval']
            
            for agent_i in range(*aii):
                agent = self._agents[agent_i]
                
                map_frame_track_id_to_index[frame_i][agent['track_id']] = agent_i
                
                if agent['track_id'] > max_track_id:
                    max_track_id = agent['track_id']

        return int(max_track_id), map_frame_track_id_to_index
    
    def _compute_moving_angular_velocity(self, track_id):
        # compute angular momentum with self._window sized window
        # max window to compute on is 100 - self._window

        ret = [self._compute_average_angular_velocity(track_id=track_id, start_frame_id=sfi)
               for sfi in range(100 - self._window)]

        # if there are any nan, replace them with with set replacement value
        for i, speed in enumerate(ret):
            ret[i] = speed if (isinstance(speed, np.float64) and not np.isnan(speed)) else self._nan_replacement

        return ret
    
    def _compute_average_angular_velocity(self, track_id, start_frame_id):
        position = self._make_position_retriever(track_id, start_frame_id)
        
        positions = [position(i) for i in range(self._window)]
        deltas = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        thetas = [angle_between(deltas[i + 1], deltas[i]) for i in range(len(deltas) - 1)]
        
        avg_theta = np.mean(thetas)
        
        # in revolutions per second
        angular_velocity = FRAMES_PER_SECOND * avg_theta / (2 * np.pi)
        
        return angular_velocity
            
            
    def _make_position_retriever(self, track_id, start_frame_id):
        def position(i):
            frame_i = self._fii[0] + start_frame_id + i
            
            # if it is not in the frame, then we do not consider it and return NaN
            if track_id not in self._map_ftiti[frame_i]:
                return np.NaN
            
            agent_i = self._map_ftiti[self._fii[0] + start_frame_id + i][track_id]
            return self._agents[agent_i]['centroid']
        
        return position