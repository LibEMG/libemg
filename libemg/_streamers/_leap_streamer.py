import asyncio
from websockets import client
import json
from multiprocessing import Process, Event, Lock
from libemg.shared_memory_manager import SharedMemoryManager
import numpy as np
import time

class LeapStreamer(Process):
    def __init__(self, shared_memory_items):
        Process.__init__(self, daemon=True)
        self.uri = "ws://localhost:6437/v7.json"
        self.shared_memory_items = shared_memory_items
        self.keys_to_collect = [i[0] for i in self.shared_memory_items if 'count' not in i[0]]
        self.data_handlers = []

    def run(self):
        self.smm = SharedMemoryManager()
        for item in self.shared_memory_items:
            self.smm.create_variable(*item)
        
        def write_key(value, key):
            self.smm.modify_variable(key, lambda x: np.vstack((value, x))[:x.shape[0],:])
            self.smm.modify_variable(key+"_count", lambda x: x + value.shape[0])
        self.data_handlers.append(write_key)

        asyncio.run(self.start_stream())


    async def start_stream(self):
        self.ws_client = await client.connect(self.uri)
        async for message in self.ws_client:
            message = message.strip()
            if ("serviceVersion" in message) or ("event" in message):
                continue
            try:
                packet = json.loads(message)
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {message}")
                continue
            
            if len(packet):
                timestamp = time.time()
                for key in self.keys_to_collect:
                    try:
                        handle = getattr(self, "get_" + key)
                        data   = handle(packet)
                        if len(data):
                            timestamp_ = np.ones((data.shape[0], 1))*timestamp
                            data   = np.hstack((timestamp_, data))
                            for h in self.data_handlers:
                                h(data, key)
                    except Exception as e:
                        print(f"Error occurred obtaining {key}, \n{e}")

    def get_arm_basis(self, data):
        ret = []
        for h in data['hands']:
            ret.append(np.hstack(([h['id']]+h['armBasis'][:])))
        return np.array(ret)
    
    def get_arm_width(self, data):
        ret = []
        for h in data['hands']:
            ret.append([h['id'], h['armWidth']])
        return np.array(ret)
    
    def get_hand_direction(self, data):
        ret = []
        for h in data['hands']:
            ret.append([h['id']]+h['direction'])
        return np.array(ret)
    
    def get_elbow(self, data):
        ret = []
        for h in data['hands']:
            ret.append([h['id']]+h['elbow'])
        return np.array(ret)

    def get_grab_angle(self, data):
        ret = []
        for h in data['hands']:
            ret.append([h['id'],h['grabAngle']])
        return np.array(ret)

    def get_grab_strength(self, data):
        ret = []
        for h in data['hands']:
            ret.append([h['id'], h['grabStrength']])
        return np.array(ret)
    
    def get_palm_normal(self, data):
        ret = []
        for h in data['hands']:
            ret.append([h['id']]+h['palmNormal'])
        return np.array(ret)

    def get_palm_position(self, data):
        ret = []
        for h in data['hands']:
            ret.append([h['id']]+h['palmPosition'])
        return np.array(ret)
    
    def get_palm_velocity(self, data):
        ret = []
        for h in data['hands']:
            ret.append([h['id']]+h['palmVelocity'])
        return np.array(ret)

    def get_palm_width(self, data):
        ret = []
        for h in data['hands']:
            ret.append([h['id'], h['palmWidth']])
        return np.array(ret)

    def get_pinch_distance(self, data):
        ret = []
        for h in data['hands']:
            ret.append([h['id'], h['pinchDistance']])
        return np.array(ret)
    
    def get_pinch_strength(self, data):
        ret = []
        for h in data['hands']:
            ret.append([h['id'], h['pinchStrength']])
        return np.array(ret)
    
    def get_handedness(self, data):
        ret = []
        for h in data['hands']:
            if h['type'] == 'right':
                val = 0
            else:
                val = 1
            ret.append([h['id'],val])
        return np.array(ret)
    
    def get_hand_r(self, data):
        ret = []
        for h in data['hands']:
            ret.append(np.hstack(([h['id']]+h['r'][:])))
        return np.array(ret)
    
    def get_hand_s(self, data):
        ret = []
        for h in data['hands']:
            ret.append([h['id'],h['s']])
        return np.array(ret)

    def get_sphere_center(self, data):
        ret = []
        for h in data['hands']:
            ret.append([h['id']]+h['sphereCenter'])
        return np.array(ret)

    def get_sphere_radius(self, data):
        ret = []
        for h in data['hands']:
            ret.append([h['id'],h['sphereRadius']])
        return np.array(ret)
    
    def get_wrist(self, data):
        ret = []
        for h in data['hands']:
            ret.append([h['id']]+h['wrist'])
        return np.array(ret)


    # <--- Above = Hand Info ---> #
    # <--- Below = Finger Info ---> #

    def get_finger_bases(self, data):
        ret = []
        for p in data['pointables']:
            ret.append([p['id']]+ np.array(p['bases']).flatten().tolist())
        return np.array(ret)

    def get_btip_position(self, data):
        ret = []
        for p in data['pointables']:
            ret.append([p['id']] + p['btipPosition'])
        return np.array(ret)

    def get_carp_position(self, data):
        ret = []
        for p in data['pointables']:
            ret.append([p['id']] + p['carpPosition'])
        return np.array(ret)
    
    def get_dip_position(self, data):
        ret = []
        for p in data['pointables']:
            ret.append([p['id']] + p['dipPosition'])
        return np.array(ret)
    
    def get_finger_direction(self, data):
        ret = []
        for p in data['pointables']:
            ret.append([p['id']] + p['direction'])
        return np.array(ret)
    
    def get_finger_extended(self, data):
        ret = []
        for p in data['pointables']:
            ret.append([p['id'], int(p['extended'])])
        return np.array(ret)
    
    def get_finger_length(self, data):
        ret = []
        for p in data['pointables']:
            ret.append([p['id'], p['length']])
        return np.array(ret)
    
    def get_mcp_position(self, data):
        ret = []
        for p in data['pointables']:
            ret.append([p['id']] + p['mcpPosition'])
        return np.array(ret)
    
    def get_pip_position(self, data):
        ret = []
        for p in data['pointables']:
            ret.append([p['id']] + p['pipPosition'])
        return np.array(ret)
    
    def get_stabilized_tip_position(self, data):
        ret = []
        for p in data['pointables']:
            ret.append([p['id']] + p['stabilizedTipPosition'])
        return np.array(ret)
    
    def get_tip_position(self, data):
        ret = []
        for p in data['pointables']:
            ret.append([p['id']]+p['tipPosition'])
        return np.array(ret)
    
    def get_tip_velocity(self, data):
        ret = []
        for p in data['pointables']:
            ret.append([p['id']] + p['tipVelocity'])
        return np.array(ret)

    def get_tool(self, data):
        ret = []
        for p in data['pointables']:
            ret.append([p['id'], int(p['tool'])])
        return np.array(ret)
    
    def get_touch_distance(self, data):
        ret = []
        for p in data['pointables']:
            ret.append([p['id'], p['touchDistance']])
        return np.array(ret)
    
    def get_touch_zone(self, data):
        ret = []
        for p in data['pointables']:
            if p['touchZone'] == 'none':
                val = 0
            elif p['touchZone'] == 'hovering':
                val = 1
            elif p['touchZone'] == 'touching':
                val = 2
            else: # invalid?
                val = -1
            ret.append([p['id'],val])
        return np.array(ret)
    
    def get_finger_width(self, data):
        ret = []
        for p in data['pointables']:
            ret.append([p['id'], p['width']])
        return np.array(ret)

if __name__ == "__main__":
    ls = LeapStreamer()
    asyncio.run(ls.start_stream())
    # asyncio.run(main())