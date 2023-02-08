from enum import unique
from re import U
import sapien.core as sapien
from sapien.utils import Viewer
import numpy as np
from soupsieve import select
import trimesh
import open3d
import os
import ipdb
import torch
from pointnet2_ops import pointnet2_utils

def mkdir(path):
    # 引入模块
    import os
 
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path) 
 
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return False
 


class PC_Generator:
    def __init__(self):
        pass

    def get_part_mesh(self, link, global_transform=True):
        final_vs = [];
        final_fs = [];
        vid = 0;
        vs = []
        for s in link.get_collision_shapes():
            v = np.array(s.geometry.vertices, dtype=np.float32)
            f = np.array(s.geometry.indices, dtype=np.uint32).reshape(-1, 3)
            vscale = s.geometry.scale
            v[:, 0] *= vscale[0];
            v[:, 1] *= vscale[1];
            v[:, 2] *= vscale[2];
            ones = np.ones((v.shape[0], 1), dtype=np.float32)
            v_ones = np.concatenate([v, ones], axis=1)
            pose = s.get_local_pose()
            transmat = pose.to_transformation_matrix()
            v = (v_ones @ transmat.T)[:, :3]
            vs.append(v)
            final_fs.append(f + vid)
            vid += v.shape[0]
        part_transmat = None
        if len(vs) > 0:
            vs = np.concatenate(vs, axis=0)
            part_transmat = link.get_pose().to_transformation_matrix()
            if global_transform :
                ones = np.ones((vs.shape[0], 1), dtype=np.float32)
                vs_ones = np.concatenate([vs, ones], axis=1)
                vs = (vs_ones @ part_transmat.T)[:, :3]
            final_vs.append(vs)
        if(final_fs!=[] and final_fs!=[]):
            final_vs = np.concatenate(final_vs, axis=0)
            final_fs = np.concatenate(final_fs, axis=0)
        return final_vs, final_fs, part_transmat


    def sample_pc(self, v, f, n_points, camera_sample=False):
        mesh = trimesh.Trimesh(vertices=v, faces=f)
        points, __ = trimesh.sample.sample_surface(mesh=mesh, count=n_points)

        # open3d.visualization.draw_geometries([pcd])
        return np.asarray(points)
    
    def remove_hidden(self, points) :

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)

        diameter = np.linalg.norm(
            np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))

        camera_1 = [-diameter, 0, diameter]
        camera_2 = [-diameter, diameter, diameter]
        camera_3 = [-diameter, -diameter, diameter]
        radius = diameter*100

        _, pt_map_1 = pcd.hidden_point_removal(camera_1, radius)
        _, pt_map_2 = pcd.hidden_point_removal(camera_2, radius)
        _, pt_map_3 = pcd.hidden_point_removal(camera_3, radius)

        idx = np.concatenate([pt_map_1, pt_map_2, pt_map_3])
        idx = np.unique(idx)

        return pcd, idx
    
    def transform(self, vs, mat) :

        ones = np.ones((vs.shape[0], 1), dtype=np.float32)
        vs_ones = np.concatenate([vs, ones], axis=1)
        vs = (vs_ones @ mat.T)[:, :3]

        return vs
    
    def uniform_distance_on_two_sets(self, vs1, vs2, n_points) :

        pc_bias = np.array([[100, 100, 100]])

        merged_pc = np.concatenate([vs1+pc_bias, vs2])
        merged_pc_tensor = torch.tensor(merged_pc, device="cuda").view(1, -1, 3).float()
        vs1_size = vs1.shape[0]

        selected_point_id = pointnet2_utils.furthest_point_sample(merged_pc_tensor, n_points).long()[0]

        selected_point_id = np.asarray(selected_point_id.cpu())

        # transform moving part back to its original origin
        vs1_idx = selected_point_id[selected_point_id<vs1_size]
        vs2_idx = selected_point_id[selected_point_id>=vs1_size] - vs1_size

        return vs1_idx, vs2_idx
    
    def sample_static(self, src_path, dst_path, n_points) :

        mkdir(dst_path)

        engine = sapien.Engine()
        renderer = sapien.VulkanRenderer()
        engine.set_renderer(renderer)

        scene_config = sapien.SceneConfig()
        scene = engine.create_scene(scene_config)
        scene.set_timestep(1 / 240.0)
        scene.add_ground(0)

        rscene = scene.get_renderer_scene()
        rscene.set_ambient_light([0.5, 0.5, 0.5])
        rscene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        loader: sapien.URDFLoader = scene.create_urdf_loader()
        loader.fix_root_link = True
        robot: sapien.Articulation = loader.load(src_path)
        robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        links = robot.get_links()
        robot.set_qpos(np.zeros((robot.dof)))

        v, f, mat, id = [], [], [], 0

        pc = []
        
        for link in links:

            cur_v, cur_f, cur_mat = self.get_part_mesh(link, global_transform = True)
            if len(cur_v) :
                v.append(cur_v)
                f.append(cur_f + id)
                mat.append(cur_mat)
                id += len(cur_v)
        
        # get mesh for moving part and fixed part
        v = np.concatenate(v)
        f = np.concatenate(f)
        mat = np.concatenate(mat)

        points = self.sample_pc(v, f, n_points*4)
        point_tensor = torch.tensor(points, device="cuda").float().contiguous()

        selected_point_id = pointnet2_utils.furthest_point_sample(point_tensor.view(1, -1, 3), n_points).long()[0]
        selected_pc = point_tensor[selected_point_id]
        
        torch.save(selected_pc.to("cpu"), os.path.join(dst_path, "pointcloud_tensor"))

    def demo(self, src_path, dst_path, moving_part="none", n_points=1024):

        mkdir(dst_path)

        engine = sapien.Engine()
        renderer = sapien.VulkanRenderer()
        engine.set_renderer(renderer)

        scene_config = sapien.SceneConfig()
        scene = engine.create_scene(scene_config)
        scene.set_timestep(1 / 240.0)
        scene.add_ground(0)

        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        # viewer = Viewer(renderer)
        # viewer.set_scene(scene)
        # viewer.set_camera_xyz(x=-2, y=0, z=1)
        # viewer.set_camera_rpy(r=0, p=-0.3, y=0)

        loader: sapien.URDFLoader = scene.create_urdf_loader()
        loader.fix_root_link = True
        robot: sapien.Articulation = loader.load(src_path)
        robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        links = robot.get_links()
        moving_link = None
        
        for link_i in links:
            if moving_part == link_i.name:
                moving_link = link_i
        
        assert(moving_link != None)

        robot.set_qpos(np.zeros((robot.dof)))

        moving_v, moving_f, moving_mat, moving_id = [], [], [], 0
        fixed_v, fixed_f, fixed_id = [], [], 0

        moving_pc = []
        fixed_pc = []
        
        for link in links:

            if link.name == moving_link.name :
                v, f, trs_mat = self.get_part_mesh(link, global_transform = False)
                if len(v) :
                    moving_v.append(v)
                    moving_f.append(f + moving_id)
                    moving_mat.append(trs_mat)
                    moving_id += len(v)
            else :
                v, f, trs_mat = self.get_part_mesh(link, global_transform = True)
                if len(v) :
                    fixed_v.append(v)
                    fixed_f.append(f + fixed_id)
                    fixed_id += len(v)
        
        # get mesh for moving part and fixed part
        moving_v = np.concatenate(moving_v)
        moving_f = np.concatenate(moving_f)
        moving_mat = np.concatenate(moving_mat)
        fixed_v = np.concatenate(fixed_v)
        fixed_f = np.concatenate(fixed_f)

        # sample a dense pointcloud for each part
        moving_pc = self.sample_pc(moving_v, moving_f, n_points*32)
        moving_pc_trs = self.transform(moving_pc, moving_mat)
        fixed_pc = self.sample_pc(fixed_v, fixed_f, n_points*32)

        # sample uniform points on two parts
        moving_pc_idx, fixed_pc_idx = self.uniform_distance_on_two_sets(moving_pc_trs, fixed_pc, n_points*8)
        moving_pc_trs = moving_pc_trs[moving_pc_idx]
        moving_pc = moving_pc[moving_pc_idx]
        fixed_pc = fixed_pc[fixed_pc_idx]
        moving_pc_size = moving_pc_trs.shape[0]

        # select visible points from the two parts
        pcd, visible_idx = self.remove_hidden(np.concatenate([moving_pc_trs, fixed_pc]))

        # remove hidden points from moving_pc and fixed pc
        moving_pc = moving_pc[visible_idx[visible_idx < moving_pc_size]]

        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(moving_pc)
        # open3d.visualization.draw_geometries([pcd])

        fixed_pc = fixed_pc[visible_idx[visible_idx >= moving_pc_size] - moving_pc_size]

        # down sample to proper size on both parts
        moving_pc_idx, fixed_pc_idx = self.uniform_distance_on_two_sets(moving_pc, fixed_pc, n_points)
        moving_pc = moving_pc[moving_pc_idx]
        fixed_pc = fixed_pc[fixed_pc_idx]
        moving_pc = torch.tensor(moving_pc).float()
        fixed_pc = torch.tensor(fixed_pc).float()

        # append masks
        moving_pc = torch.cat((moving_pc, torch.ones((moving_pc.shape[0], 1))), dim=-1)
        fixed_pc = torch.cat((fixed_pc, torch.zeros((fixed_pc.shape[0], 1))), dim=-1)
        
        # merge the parts
        selected_pc = torch.cat((moving_pc, fixed_pc), dim=0)

        # save the pointcloud
        torch.save(selected_pc, os.path.join(dst_path, "pointcloud_tensor"))

        # viewer = Viewer(renderer)  # Create a viewer (window)
        # viewer.set_scene(scene)  # Bind the viewer and the scene

        # # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
        # # The principle axis of the camera is the x-axis
        # viewer.set_camera_xyz(x=-4, y=0, z=2)
        # # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
        # # The camera now looks at the origin
        # viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
        # viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

        # while not viewer.closed:  # Press key q to quit
        #     scene.step()  # Simulate the world
        #     scene.update_render()  # Update the world to the renderer
        #     viewer.render()

        # moving_point_cloud = open3d.geometry.PointCloud()
        # fixed_point_cloud = open3d.geometry.PointCloud()
        # moving_point_cloud.points = open3d.utility.Vector3dVector(moving_pc[selected_point_id[0], :])
        # fixed_point_cloud.points = open3d.utility.Vector3dVector(fixed_pc[selected_point_id[1], :])
        # open3d.io.write_point_cloud(os.path.join(dst_path, "moving_part.ply"), moving_point_cloud, True)
        # open3d.io.write_point_cloud(os.path.join(dst_path, "fixed_part.ply"), fixed_point_cloud, True)

        # open3d.visualization.draw_geometries([moving_point_cloud, fixed_point_cloud])

            # np.save(save_path, obj_pc)
            
            # point_cloud = open3d.geometry.PointCloud()

            # point_cloud.points = open3d.utility.Vector3dVector(obj_pc)
            # open3d.io.write_point_cloud(save_path+".ply", point_cloud, True)    # 默认false，保存为Binarty；True 保存为ASICC形式
            # # open3d.visualization.draw_geometries([point_cloud])
            # vis = open3d.visualization.Visualizer()
            
            # vis.create_window()
            # vis.add_geometry(point_cloud)
            # # vis.update_geometry(point_cloud)
            # vis.poll_events()
            # vis.update_renderer()
            # # image path
            # image_path = self.save_path+str(robot.get_links()[i])+'.jpg'
            # vis.capture_screen_image(image_path)
            # vis.destroy_window()


if __name__ == '__main__':
    # folder = os.walk("assets/dataset/one_door_cabinet")

    # for path,dir_list,file_list in folder:  
    #     print(path)
        
    ## points number to sample
    n_points = 128
    ## URDF path
    URDF_path = "franka_description/robots/franka_panda_longer.urdf"
    ## the object name
    obj_name='pot2'
    ## the folder path
    save_path = "tmp/"

    generator = PC_Generator()
    generator.demo(src_path=URDF_path, dst_path=save_path, moving_part = "panda_leftfinger", n_points=n_points)
