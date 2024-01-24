import pybullet as p
import PySimpleGUI as sg
import pickle
from os import getcwd
from os.path import abspath, dirname, basename, splitext, join
import numpy as np
from pybullet_utils import get_body_pose, get_link_pose, get_link_collision_shape


class PyBulletRecorder:
    class LinkTracker:
        def __init__(self,
                     name,
                     body_id,
                     link_id,
                     link_origin,
                     mesh_type,
                     mesh_path,
                     mesh_scale,
                     client_id=0):
            self.body_id = body_id
            self.link_id = link_id
            self.mesh_type = mesh_type
            self.mesh_path = mesh_path
            self.mesh_scale = mesh_scale
            self.link_origin = link_origin
            self.name = name
            self.client_id = client_id

        def transform(self, position, orientation):
            return p.multiplyTransforms(
                position, orientation,
                self.link_origin[0], self.link_origin[1],
            )

        def get_keyframe(self):
            if self.link_id == -1:
                position, orientation = p.getBasePositionAndOrientation(
                    self.body_id, self.client_id)
                position, orientation = self.transform(
                    position=position, orientation=orientation)
            else:
                link_state = p.getLinkState(self.body_id, self.link_id,
                                            computeForwardKinematics=True)
                position, orientation = self.transform(
                    position=link_state[4],
                    orientation=link_state[5])
            return {
                'position': list(position),
                'orientation': list(orientation)
            }


    def __init__(self, client_id=0):
        self.states = []
        self.links = []
        self.client_id = client_id
        # Make sure PybulletRecorder is called in the main workspace
        self.workAbsFolderPath = getcwd()
        self.fake_pose = {
            'position': [1000, 1000, 0],
            'orientation': [0, 0, 0, 1.]
        }


    def register_object(self, body_id, body_name=None):
        link_id_map = dict()
        n = p.getNumJoints(body_id)
        link_id_map[p.getBodyInfo(body_id, physicsClientId=self.client_id)[0].decode('gb2312')] = -1
        for link_id in range(0, n):
            link_id_map[p.getJointInfo(body_id, link_id, physicsClientId=self.client_id)[
                12].decode('gb2312')] = link_id

        for linkName in link_id_map:
            link_id = link_id_map[linkName]
            link_collision_infos = get_link_collision_shape(body_id, link_id, client_id=self.client_id)
            for i, link_collision_info in enumerate(link_collision_infos):
                if len(link_collision_info) == 0:
                    print(f"No collision shape for {linkName}")
                    continue
            
                link_mesh_type, link_mesh_scale, link_mesh_path, linkjoint2linkcenter_pos, linkjoint2linkcenter_ori \
                    = link_collision_info[2:7]
                
                self.links.append(
                    PyBulletRecorder.LinkTracker(
                        name=str(body_name) + f'_bdID_{body_id}_{linkName}_{i}',
                        body_id=body_id,
                        link_id=link_id,
                        # ***Note: linkjoint2linkcenter_pos is already scaled by mesh_scale, so we don't need to scale it again!
                        link_origin=[linkjoint2linkcenter_pos, linkjoint2linkcenter_ori],
                        mesh_type=link_mesh_type,
                        mesh_path=join(self.workAbsFolderPath, link_mesh_path.decode(encoding='utf-8')),
                        mesh_scale=link_mesh_scale,
                        client_id=self.client_id
                        )
                    )


    def add_keyframe(self):
        # Ideally, call every p.stepSimulation()
        current_state = {}
        for link in self.links:
            current_state[link.name] = link.get_keyframe()
        self.states.append(current_state)


    def prompt_save(self):
        layout = [[sg.Text('Do you want to save previous episode?')],
                  [sg.Button('Yes'), sg.Button('No')]]
        window = sg.Window('PyBullet Recorder', layout)
        save = False
        while True:
            event, values = window.read()
            if event in (None, 'No'):
                break
            elif event == 'Yes':
                save = True
                break
        window.close()
        if save:
            layout = [[sg.Text('Where do you want to save it?')],
                      [sg.Text('Path'), sg.InputText(getcwd())],
                      [sg.Button('OK')]]
            window = sg.Window('PyBullet Recorder', layout)
            event, values = window.read()
            window.close()
            self.save(values[0])
        self.reset()


    def reset(self, links=True):
        self.states = []
        if links:
            self.links = []


    def get_formatted_output(self):
        retval = {}
        for link in self.links:
            retval[link.name] = {
                'type': 'mesh',
                'mesh_path': link.mesh_path,
                'mesh_scale': link.mesh_scale,
                # fake_pose (prepare area) if link not in state, it is probably that this link is registered later.
                'frames': [state.get(link.name, self.fake_pose) for state in self.states], 
            }
        return retval


    def save(self, path):
        if path is None:
            print("[Recorder] Path is None.. not saving")
        else:
            print("[Recorder] Saving state to {}".format(path))
            pickle.dump(self.get_formatted_output(), open(path, 'wb'))