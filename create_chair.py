import os
import pathlib
import xml.etree.ElementTree as ET
import shutil
import yaml
import numpy as np

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

def process_cabinet(path, scale) :

	semantic_file = os.path.join(path, "semantics.txt")
	urdf_file = os.path.join(path, "mobility.urdf")
	name = path.split('/')[-1]
	src_path = path
	dst_path = os.path.join("dataset", "chair", name)

	urdf = ET.parse(urdf_file)
	root = urdf.getroot()

	recursive_scale(root, scale)
	
	with open(semantic_file, "r") as f:

		semantics = f.read().splitlines()
		linknum = len(semantics)
	
	if not os.path.exists(dst_path):
		# 如果目标路径不存在原文件夹的话就创建
		os.makedirs(dst_path)

	if os.path.exists(src_path):
		# 如果目标路径存在原文件夹的话就先删除
		shutil.rmtree(dst_path)

	shutil.copytree(src_path, dst_path)

	urdf.write(os.path.join(dst_path, 'mobility.urdf'), xml_declaration=True)
	
	return [name], [path], [linknum]

if __name__ == "__main__" :

	root = "./assets"
	scale = 0.3
	cabinet_name_list = get_cabinet_name_list("./chair.txt")
	print("total", len(cabinet_name_list), "chairs")

	name_list = []
	path_list = []
	link_num_list = []

	for name in cabinet_name_list :
		
		new_name_list, new_path_list, new_link_num_list = process_cabinet(os.path.join(root, name), scale)
		name_list += new_name_list
		path_list += new_path_list
		link_num_list += new_link_num_list
	
	print("total", len(name_list), "new chairs")
	
	save_dict = {}
	
	for name, path, link_num in zip(name_list, path_list, link_num_list) :
		item = {
			"name": name,
			"path": os.path.join(path, "mobility.urdf"),
			"link_num": link_num
		}
		save_dict[name] = item
	
	with open("./chair_conf.yaml", "w") as f:
		print(yaml.dump(save_dict), file=f)