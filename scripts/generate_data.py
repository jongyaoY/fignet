
from robosuite.models.world import MujocoWorldBase
from robosuite.models.arenas import EmptyArena
from robosuite.utils.binding_utils import MjSim, MjRenderContext
from robosuite.utils import OpenCVRenderer
from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import (
    string_to_array,
    array_to_string, 
    find_elements, 
    )
from typing import List
import tqdm
from scipy.spatial.transform import Rotation as R
import numpy as np
from rigid_fall.mesh_object import get_all_objects, MeshObject

# Params
num_object_range = [3, 5]
ep_len = 200
total_steps = 1000
render = False
data_path = "datasets/mujoco_moviA"
data_path = data_path + "_" + str(total_steps)

camera_name = 'frontview'
arena = EmptyArena()
matrix = [[0, 1, 0],
        [-0.91997056,0., 0.39198746],
        [0.39198746, 0., 0.91997056]]
matrix = np.array(matrix).T
cam_quat = R.from_matrix(matrix).as_quat()
arena.set_camera(
    camera_name, 
    pos=[0.8, 0., 1.5], 
    quat=convert_quat(cam_quat, "wxyz")
    )


all_objects = get_all_objects()


def init_sim(arena, objects):
    mujoco_model = MujocoWorldBase()
    mujoco_model.merge(arena)
    # suite.models.assets_root
    # dyn_bodies = []
    for o in objects:
        mujoco_model.merge_assets(o)
        mujoco_model.worldbody.append(o.get_obj())
        # dyn_bodies.append(o.name)

    sim = MjSim.from_xml_string(mujoco_model.get_xml())
    render_context = MjRenderContext(sim)
    sim.add_render_context(render_context)

    viewer = OpenCVRenderer(sim)
    viewer.set_camera(sim.model.camera_name2id(camera_name))
    return sim, viewer, mujoco_model.get_xml()

def choose_objects(all_objects, num_range):
    num_obj = np.random.randint(low=num_range[0], high=num_range[1])
    num_obj = min(num_obj, len(all_objects))
    objects = np.random.choice(all_objects, num_obj, replace=False)
    return objects.tolist()

def get_object_properties(objects: List[MeshObject]):
    properties = {}
    for obj in objects:
        geom = find_elements(root=obj.get_obj(), tags="geom", attribs={"group": "0"})
        mesh_name = geom.get("mesh")
        mesh = find_elements(root=obj.asset, tags="mesh", attribs={"name": mesh_name})
        mesh_file = mesh.get("file")
        solref = string_to_array(geom.get("solref"))
        solimp = string_to_array(geom.get("solimp"))
        friction = string_to_array(geom.get("friction"))
        density = string_to_array(geom.get("density"))
        properties[obj.name] = {
            "solref": solref,
            "solimp": solimp,
            "friction": friction,
            "density": density,
            "mesh": mesh_file
        }
    return properties

def random_floor_properties():
    friction = np.random.uniform(low=0.0, high=1.0, size=(1, )).item()
    restitution = np.random.uniform(low=0.8, high=1.2, size=(1,)).item()
    floor_properties = {
    "solref": [0.001, restitution],
    "solimp": [0.998, 0.998, 0.001],
    "friction": [friction, 0.3*friction, 0.1*friction],
    "priority": [1],
    }
    return floor_properties

def set_floor_properties(arena, properties):
    floor = find_elements(root=arena.worldbody, tags="geom", attribs={"name": "floor"})
    for name, p in properties.items():
        floor.set(name, array_to_string(p))


def get_object_names(sim):
    names = []
    for name in sim.model.body_names:
        if name == "world":
            continue
        if "_" in name:
            name = name.split("_")[0]
        names.append(name)
    return names

def process_object_properties(prop):
    prop_out = {}
    for k, v in prop.items():
        if k == "priority" or k == "solimp":
            continue
        elif k == "solref":
            restitution = v[1]
            prop_out["restitution"] = restitution
        else:
            prop_out[k] = v
    return prop_out

def init_data(mujoco_xml, object_properties, object_names):
    data = {
        "obj_id": {},
        "timestep": [],
        "pos": [],
        "quat": [],
        "velp": [],
        "velr": [],
    }
    
    floor_prop = process_object_properties(object_properties["floor"])
    meta_data = {
        "env": {
            "floor": {
                "type": "box", 
                "extents": [3.0, 3.0, 0.5],
                "properties": floor_prop,
                "initial_pose":[0, 0, -0.5/2.],
            }
        },
        "objects": {}
    }
    for i, name in enumerate(object_names):
        data["obj_id"][name] = i
        obj_prop = process_object_properties(object_properties[name])
        mesh_file = obj_prop["mesh"]
        del obj_prop["mesh"]
        meta_data["objects"].update(
            {
                name: {"mesh": mesh_file, "properties": obj_prop}
            }
        )
    data["meta_data"] = meta_data
    data["mujoco_xml"] = mujoco_xml
    return data

def record_step(sim, data):
    data["timestep"].append(sim.data.time)
    pos = []
    quat = []
    velp = []
    velr = []
    for name in sim.model.body_names:
        if name == "world":
            continue
        bid = sim.model.body_name2id(name)
        pos.append(sim.data.body_xpos[bid])
        quat.append(sim.data.body_xquat[bid][[1, 2, 3, 0]])
        velp.append(sim.data.get_body_xvelp(name))
        velr.append(sim.data.get_body_xvelr(name))
    data["pos"].append(np.array(pos))
    data["quat"].append(np.array(quat))
    data["velp"].append(np.array(velp))
    data["velr"].append(np.array(velr))

    
def rollout(sim, data, steps, total_steps, viewer=None, pbar=None, render=False):
    trial = 0
    num_sample_trial = 10
    vel_scale = 0.7
    body_names = get_object_names(sim)
    # Randomize positions
    while trial < num_sample_trial:
        for _, name in enumerate(body_names):
            rand_pos = np.concatenate(
                    [
                        np.random.uniform(-0.15, 0.15, 2), # xy pos
                        np.random.uniform(0.1, 0.15, 1) # z pos
                    ]
                )
            rand_quat = R.random().as_quat()[[3, 0, 1, 2]]
            rand_pose = np.concatenate([rand_pos, rand_quat], axis=-1)
            joint_name = name + "_joint0"
            sim.data.set_joint_qpos(joint_name, rand_pose)
        sim.forward()
        if sim.data.ncon == 0:
            break
        trial+=1
    # Randomize velocities
    rand_vel = np.random.randn(sim.data.qvel.shape[0]) * vel_scale
    sim.data.qvel = rand_vel
    # Rollout loop
    for t in range(ep_len):
        sim.forward()
        sim.step()
        record_step(sim, data)
        if render and viewer is not None:
            viewer.render()
        
        if pbar is not None:
            pbar.update(1)
        if steps + t > total_steps:
            return

    # Transform results
    for k, v in data.items():
        data[k] = np.asarray(v)
    if data[k].dtype == np.float64:
        if np.isnan(data[k]).all():
            raise RuntimeError("nan")

if __name__ == "__main__":

    properties = get_object_properties(all_objects)
    data_storage = []
    steps = 0

    pbar = tqdm.tqdm(total=total_steps, desc="Sampling rollouts")
    while steps < total_steps:
        floor_properties = random_floor_properties()
        set_floor_properties(arena, floor_properties)
        properties.update(
            {"floor": floor_properties}
        )
        rand_objs = choose_objects(all_objects, num_object_range)
        sim, viewer, mujoco_xml = init_sim(arena, rand_objs)
        body_names = get_object_names(sim)
        data = init_data(mujoco_xml, properties, body_names)
        # Generate one episode
        rollout(sim, data, steps, total_steps, viewer, pbar, render)
        steps += data['pos'].shape[0]
        data_storage.append(data)
    pbar.close()
    print(
        f"collected {len(data_storage)} episodes, \
            {sum([data['timestep'].shape[0] for data in data_storage])} steps"
    )
    np.savez(data_path, data_storage)

# for i in range(physics.model.nbody):
#     body_name = physics.model.id2name(i, mujoco.mjtObj.mjOBJ_BODY)
#     body_name = body_name.split("_")[0]
#     if body_name != 'world':
#         dyn_bodies.append(body_name)
# steps = 0
# while steps < total_steps:
#     # Sample one episode
#     with physics.reset_context():
#         # Randomize positions
#         trial = 0
#         while trial < num_sample_trial:
#             for i, name in enumerate(dyn_bodies):
#                 pose_name = name +'_joint0'
#                 rand_pos = np.concatenate(
#                     [
#                         np.random.uniform(-0.15, 0.15, 2),
#                         np.random.uniform(0.1, 0.15, 1)
#                     ]
#                 )
#                 rand_quat = R.random().as_quat()[[3, 0, 1, 2]]
#                 rand_pose = np.concatenate([rand_pos, rand_quat], axis=-1)
#                 physics.named.data.qpos[pose_name] = rand_pose
#             physics.forward()
#             if physics.data.ncon == 0:
#                 break
#             else:
#                 print("Collision, resample")
#         # Randomize velocities
#         rand_vel = np.random.randn(len(dyn_bodies) * 6) * 0.8
#         physics.data.qvel = rand_vel
#     # Generate rollout
#     for t in range(ep_len):
#         physics.step()
#         if render:
#             im = physics.render(
#                 height=800,
#                 width=1024,
#                 camera_id=0
#             )
#             cv2.imshow("offscreen render", im)
#             cv2.waitKey(1)
#     steps += 1