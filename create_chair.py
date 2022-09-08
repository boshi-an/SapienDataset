import json
import os
import pathlib
import xml.etree.ElementTree as ET
import shutil
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import numpy as np

from PC_Generator import PC_Generator

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

def recursive_fix(root) :

    if root.tag == "joint" :
        root.attrib['type'] = "fixed"

    for son in root :
        recursive_fix(son)

pc_generator = PC_Generator()

def build_new_cabinet(source_path, dst_path, scale) :

    bounding_box_file = os.path.join(source_path, "bounding_box.json")
    urdf_file = os.path.join(source_path, "mobility.urdf")
    name = os.path.basename(source_path)

    urdf = ET.parse(urdf_file)
    root = urdf.getroot()

    recursive_scale(root, scale)
    recursive_fix(root)
    
    with open(bounding_box_file, "r") as f :
        bounding_box = yaml.load(f, Loader=Loader)
        bounding_box["min"] = [bounding_box["min"][0]*scale, bounding_box["min"][1]*scale, bounding_box["min"][2]*scale]
        bounding_box["max"] = [bounding_box["max"][0]*scale, bounding_box["max"][1]*scale, bounding_box["max"][2]*scale]
    
    if not os.path.exists(dst_path):
        # 如果目标路径不存在原文件夹的话就创建
        os.makedirs(dst_path)

    if os.path.exists(source_path):
        # 如果目标路径存在原文件夹的话就先删除
        shutil.rmtree(dst_path)

    shutil.copytree(source_path, dst_path)

    urdf.write(os.path.join(dst_path, 'mobility.urdf'), xml_declaration=True)

    with open(os.path.join(dst_path, 'bounding_box.json'), "w") as f :
        json.dump(bounding_box, f)
    
    return

def process_cabinet(source_path, target_path, scale, point_sample, selected_names) :

    name_list = []
    path_list = []
    
    semantic_file = os.path.join(source_path, "semantics.txt")
    urdf_file = os.path.join(source_path, "mobility.urdf")
    
    with open(semantic_file, "r") as f:

        semantics = f.read().splitlines()
        new_cabinet_name = os.path.basename(source_path)
        if int(new_cabinet_name) not in selected_names :
            return [], []
        dst_path = os.path.join(target_path, new_cabinet_name)
        build_new_cabinet(source_path, dst_path, scale)
        name_list.append(new_cabinet_name)
        path_list.append(dst_path)
        pc_generator.sample_static(
            os.path.join(dst_path, "mobility.urdf"),
            os.path.join(dst_path, "point_clouds"),
            n_points = point_sample
        )
    
    return name_list, path_list

if __name__ == "__main__" :

    root = "./assets"
    target_path = "dataset/chair"
    scale = 0.3
    point_sample = 8192
    cabinet_name_list = get_cabinet_name_list("./chair.txt")
    with open("selected_chair.yaml", "r") as f :
        selected_names = yaml.load(f, Loader=Loader)
    print("total", len(cabinet_name_list), "chairs")

    name_list = []
    path_list = []

    for name in cabinet_name_list :
        
        new_name_list, new_path_list = process_cabinet(os.path.join(root, name), target_path, scale, point_sample, selected_names)
        name_list += new_name_list
        path_list += new_path_list
    
    print("total", len(name_list), "new chairs")
    
    save_dict = {}
    
    for name, path in zip(name_list, path_list) :
        item = {
            "name": name,
            "path": os.path.join(path, "mobility.urdf"),
            "boundingBox": os.path.join(path, "bounding_box.json")
        }
        save_dict[name] = item
    
    with open("./chair_conf.yaml", "w") as f:
        print(yaml.dump(save_dict), file=f)