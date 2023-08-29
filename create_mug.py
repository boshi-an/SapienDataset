import os
import xml.etree.ElementTree as ET
import shutil
import open3d as o3d
import json

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def recursive_scale(root, scale) :

    if root.tag == "origin" :
        x, y, z = root.attrib['xyz'].split(' ')
        root.attrib['xyz'] = "{} {} {}".format(float(x)*scale, float(y)*scale, float(z)*scale)
    if root.tag == "mesh" :
        root.attrib['scale'] = "{} {} {}".format(scale, scale, scale)

    for son in root :
        recursive_scale(son, scale)

def build_new_mug(src_path, dst_path, scale) :

    urdf = ET.parse(os.path.join(src_path, 'mobility.urdf'))
    root = urdf.getroot()

    recursive_scale(root, scale)

    link_name = "link_" + os.path.basename(src_path)

    for child in root :
        if child.tag == "link" :
            child.attrib["name"] = link_name
        for gchild in child :
            if gchild.tag == "visual" :
                gchild.attrib["name"] = "lid"

    new_base = ET.Element("link")
    new_base.attrib["name"] = "base"

    new_joint = ET.Element("joint")
    new_joint.attrib["name"] = "base_joint"
    new_joint.attrib["type"] = "prismatic"
    new_joint.insert(0, ET.fromstring("<parent link=\"base\"/>"))
    new_joint.insert(0, ET.fromstring(f"<child link=\"{link_name}\"/>"))
    new_joint.insert(0, ET.fromstring("<axis xyz=\"0 0 1\"/>"))
    new_joint.insert(0, ET.fromstring("<origin xyz=\"0 0 0\"/>"))
    new_joint.insert(0, ET.fromstring("<limit lower=\"0\" upper=\"0.1\"/>"))

    root.append(new_base)
    root.append(new_joint)

    if not os.path.exists(dst_path):
        # 如果目标路径不存在原文件夹的话就创建
        os.makedirs(dst_path)

    if os.path.exists(src_path):
        # 如果目标路径存在原文件夹的话就先删除
        shutil.rmtree(dst_path)

    shutil.copytree(src_path, dst_path)

    indent(root)

    urdf.write(os.path.join(dst_path, 'mobility.urdf'), xml_declaration=True)

    # Generate Bounding Box

    mesh_file_name = "mesh.stl"
    bbox_file_name = "bounding_box.json"
    mesh_data = o3d.io.read_triangle_mesh(os.path.join(src_path, mesh_file_name))
    door_bounding_box = mesh_data.get_axis_aligned_bounding_box()
    
    bbox = {
        "min": list(door_bounding_box.min_bound),
        "max": list(door_bounding_box.max_bound)
    }
    json.dump(bbox, open(os.path.join(dst_path, bbox_file_name), 'w'))

if __name__ == "__main__" :

    root = "./mug"
    target_path = "dataset/mugs"

    name_list = [
        str(a) for a in range(83)
    ]

    for name in name_list :
        build_new_mug(
            os.path.join(root, name),
            os.path.join(target_path, name+"_link_"+name),
            0.1
        )