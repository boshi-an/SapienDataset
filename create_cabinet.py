import os
import pathlib
import xml.etree.ElementTree as ET
import shutil
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import open3d as o3d
import numpy as np
from PC_Generator import PC_Generator
from tqdm import tqdm

def get_cabinet_name_list(file_name) :

    with open(file_name, "r") as f:
        
        data = f.read().splitlines()

    return data

def recursive_scale(root, scale) :

    if root.tag == "origin" :
        x, y, z = root.attrib['xyz'].split(' ')
        root.attrib['xyz'] = "{} {} {}".format(float(x)*scale, float(y)*scale, float(z)*scale)
    if root.tag == "mesh" :
        root.attrib['scale'] = "{} {} {}".format(scale, scale, scale)

    for son in root :
        recursive_scale(son, scale)

def build_new_cabinet(src_path, dst_path, link_id, scale) :

    urdf = ET.parse(os.path.join(src_path, 'mobility.urdf'))
    root = urdf.getroot()

    friction = ET.fromstring("<surface><friction><ode><mu>100.0</mu><mu2>100.0</mu2></ode></friction></surface>")

    recursive_scale(root, scale)

    door_open_dir = 1
    for child in root :
        if child.tag == "joint" :
            name = child.attrib["name"]
            vis = False
            for tmp in child.findall('child') :
                if tmp.attrib['link'] == link_id :
                    vis = True
            for tmp in child.findall('parent') :
                if tmp.attrib['link'] == link_id :
                    vis = True
            if vis :
                # the joint meant to be revolute
                for tmp in child.findall('axis') :
                    x, y ,z = tmp.attrib['xyz'].split(' ')
                    door_open_dir = int(round(float(y)))
            else :
                # not the joint meant to be revolute
                child.attrib["type"] = "fixed"
    has_handle = False
    handle_dx = 0
    handle_dy = 0
    handle_dz = 0
    handle_center = []
    door_dx = 0
    door_dy = 0
    door_dz = 0
    door_min = []
    door_max = []
    for child in root :
        if child.tag == "link" and child.attrib['name']==link_id :
            mesh_file_name_list = []
            for tmp in child.findall('visual') :
                if 'handle' in tmp.attrib['name'] :
                    has_handle = True
                    for ttmp in tmp.findall('origin') :
                        handle_dx, handle_dy, handle_dz = ttmp.attrib['xyz'].split(' ')
                    for mesh in tmp.findall('geometry') :
                        for mesh_file in mesh.findall('mesh') :
                            mesh_file_name = mesh_file.attrib['filename']
                            mesh_file_name_list.append(mesh_file_name)
                            mesh_data = o3d.io.read_triangle_mesh(os.path.join(src_path, mesh_file_name))
                            handle_center.append(mesh_data.get_center())
            for tmp in child.findall('collision') :
                is_handle = False
                for ttmp in tmp.findall('origin') :
                    door_dx, door_dy, door_dz = ttmp.attrib['xyz'].split(' ')
                for mesh in tmp.findall('geometry') :
                    for mesh_file in mesh.findall('mesh') :
                        mesh_file_name = mesh_file.attrib['filename']
                        if mesh_file_name in mesh_file_name_list :
                            is_handle = True
                        mesh_data = o3d.io.read_triangle_mesh(os.path.join(src_path, mesh_file_name))
                        door_bounding_box = mesh_data.get_axis_aligned_bounding_box()
                        door_min.append(door_bounding_box.min_bound)
                        door_max.append(door_bounding_box.max_bound)
                if is_handle :
                    tmp.append(friction)

    if has_handle :
        handle_center = np.array(handle_center).mean(axis=0) * scale
    else :
        handle_center = np.array([0,0,0])

    door_min = np.array(door_min).min(axis=0) * scale
    door_max = np.array(door_max).max(axis=0) * scale

    if not os.path.exists(dst_path):
        # 如果目标路径不存在原文件夹的话就创建
        os.makedirs(dst_path)

    if os.path.exists(src_path):
        # 如果目标路径存在原文件夹的话就先删除
        shutil.rmtree(dst_path)

    shutil.copytree(src_path, dst_path)

    urdf.write(os.path.join(dst_path, 'mobility.urdf'), xml_declaration=True)

    # do extreme simplification
    urdf_simp = urdf
    root = urdf_simp.getroot()

    for child in root :

        if child.tag == "link" and child.attrib['name']==link_id :
            # this is the link we want
            pass
        
        elif child.tag == "link" and child.attrib['name']!=link_id :
            # remove all meshes
            for visual_node in child.findall('visual') :
                child.remove(visual_node)
            for collision_node in child.findall('collision') :
                child.remove(collision_node)
    
    urdf.write(os.path.join(dst_path, 'mobility_simp.urdf'), xml_declaration=True)

    with open(os.path.join(dst_path, 'handle.yaml'), 'w') as f :

        handle_dict = {
            "has_handle": has_handle,
            "pos": {
                "x": float(handle_dx)+handle_center.tolist()[0],
                "y": float(handle_dy)+handle_center.tolist()[1],
                "z": float(handle_dz)+handle_center.tolist()[2]
            }
        }
        yaml.dump(handle_dict, f)
    
    with open(os.path.join(dst_path, 'door.yaml'), 'w') as f :

        door_dict = {
            "open_dir": door_open_dir,
            "bounding_box": {
                "xmin": float(door_dx)+door_min.tolist()[0],
                "xmax": float(door_dx)+door_max.tolist()[0],
                "ymin": float(door_dy)+door_min.tolist()[1],
                "ymax": float(door_dy)+door_max.tolist()[1],
                "zmin": float(door_dz)+door_min.tolist()[2],
                "zmax": float(door_dz)+door_max.tolist()[2]
            }
        }
        yaml.dump(door_dict, f)
    
    return has_handle

pc_generator = PC_Generator()

def process_cabinet(source_path, target_path, scale, point_sample, available_set) :

    name_list = []
    path_list = []
    total_handle = 0
    
    semantic_file = os.path.join(source_path, "semantics.txt")
    urdf_file = os.path.join(source_path, "mobility.urdf")

    with open(semantic_file, "r") as f:

        semantics = f.read().splitlines()
        for link_repr in semantics :
            link_id, link_type, link_name = link_repr.split(' ')

            if link_type == "hinge" and link_name == "rotation_door":
                # found a rotation door
                new_cabinet_name = os.path.basename(source_path) + "_"  + link_id
                if new_cabinet_name not in available_set :
                    continue
                dst_path = os.path.join(target_path, new_cabinet_name)
                total_handle += build_new_cabinet(source_path, dst_path, link_id, scale)
                name_list.append(new_cabinet_name)
                path_list.append(dst_path)
                pc_generator.demo(
                    os.path.join(dst_path, "mobility.urdf"),
                    os.path.join(dst_path, "point_clouds"),
                    moving_part = link_id,
                    n_points = point_sample
                )

    return name_list, path_list, total_handle

if __name__ == "__main__" :

    root = "./SapienDoor"
    target_path = "dataset/one_door_cabinet_maniskill"
    scale = 0.7
    point_sample = 8192
    cabinet_name_list = get_cabinet_name_list("./cabinet.txt")
    with open("selected_cabinet.yaml", "r") as f :
        selected_names = yaml.load(f, Loader=Loader)
    print("total", len(cabinet_name_list), "cabinets")

    name_list = []
    path_list = []
    total_handle = 0

    total_len = len(cabinet_name_list)

    with tqdm(total=total_len) as pbar:
        pbar.set_description('Creating Cabinets:')

        for name in cabinet_name_list :
            
            new_name_list, new_path_list, new_handle = process_cabinet(os.path.join(root, name), target_path, scale, point_sample, selected_names)
            name_list += new_name_list
            path_list += new_path_list
            total_handle += new_handle
            pbar.update(1)

    
    print("total", len(name_list), "new cabinets", total_handle, "handles")
    
    save_dict = {}
    
    for name, path in zip(name_list, path_list) :

        item = {
            "name": name,
            "path": os.path.join(path, "mobility.urdf"),
            "boundingBox": os.path.join(path, "bounding_box.json"),
            "handle": os.path.join(path, "handle.yaml"),
            "door": os.path.join(path, "door.yaml")
        }
        save_dict[name] = item
    
    with open("./cabinet_conf.yaml", "w") as f:
        print(yaml.dump(save_dict), file=f)