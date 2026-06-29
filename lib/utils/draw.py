from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os.path as osp
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import sys
from easydict import EasyDict as edict
import yaml
from models.layers.smpl.SMPL import SMPL_layer
from termcolor import colored
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import trimesh
from functools import lru_cache


children = [[1, 4, 7], [2], [3], [], [5], [6], [], [8], [9, 11, 14],
                    [10], [], [12], [13], [], [15], [16], []]
flip_pairs = np.array([[1, 2], [4, 5], [7, 8], [10, 11], [13, 14], [16, 17], [18, 19], [20, 21], [22, 23], [25, 26], [27, 28]])
joints_left=flip_pairs[:,1]
joints_right=flip_pairs[:,0]
model_path =r'data/smpl/SMPL_NEUTRAL.pkl'
with open(model_path, 'rb') as smpl_file:
    db = pickle.load(smpl_file,encoding='latin1')
cleft= '#00BFFF'
cright='#FFE07D'
cmid= '#7FFFAA'

def load_yaml(path):
    with open(path,'rb') as fid:
        cfg = yaml.safe_load(fid)
    cfg = edict(cfg)
    return cfg

def load_smpl(gender):
    h36m_jregressor = np.load('lib/model_files/J_regressor_h36m.npy')
    smpl = SMPL_layer(
                'lib/model_files/SMPL_{}.pkl'.format(gender),
                h36m_jregressor=h36m_jregressor,
                dtype=torch.float32
            ) 
    return smpl
    
def get_output(root):
    output = {}
    name_list = os.listdir(root)
    for name in name_list:
        path = os.path.join(root,name)
        output[name.split('.')[0]] = np.load(path)
    return output

def render_mesh(height, width, meshes, face, cam_param):
    # renderer
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
   
    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # camera
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # mesh
    for mesh in meshes:
        mesh = trimesh.Trimesh(mesh, face)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        # mesh.apply_transform(rot)

        # Manually apply the rotation to each vertex
        vertices_homogeneous = np.hstack((mesh.vertices, np.ones((mesh.vertices.shape[0], 1))))
        transformed_vertices = (vertices_homogeneous @ rot.T)[:, :3]
        mesh.vertices = transformed_vertices

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
        # mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene.add(mesh, 'mesh')

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    renderer.delete()
    return rgb, depth

# 固定的 180° 绕 x 旋转矩阵；用向量化比堆齐/矩阵乘快
_ROT180_X = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float32)

class _RendererSession:
    def __init__(self, width, height):
        # 仅初始化一次 renderer & scene & 灯光
        self.scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
        self.renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
        self.material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0)          # White Baseline
            # baseColorFactor=(0.7, 0.9, 1.0, 1.0)            # Color light blue  Ours
        )

        # 灯光一次性加入
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        lp = np.eye(4, dtype=np.float32)
        for v in [(0,-1,1),(0,1,1),(1,1,2)]:
            lp[:3,3] = np.array(v, dtype=np.float32)
            self.scene.add(light, pose=lp.copy())
        # 可复用：相机 node 句柄（焦距变化时更新）
        self.cam_node = None

    def set_camera(self, focal, princpt):
        cam = pyrender.IntrinsicsCamera(fx=float(focal[0]), fy=float(focal[1]),
                                        cx=float(princpt[0]), cy=float(princpt[1]))
        if self.cam_node is None:
            self.cam_node = self.scene.add(cam)
        else:
            # 直接替换 node 的 camera（pyrender 支持）
            self.scene._remove_node(self.cam_node)   # 更稳妥：先移除再添加
            self.cam_node = self.scene.add(cam)

    def render_meshes(self, meshes_xyz_list, faces, smooth=False):
        # 把若干 mesh 加入 scene，渲染后统一移除，避免每帧重建 Scene/Renderer
        nodes = []
        try:
            for V in meshes_xyz_list:
                # 轻量构建 Trimesh：process=False 关掉几何修复
                # 应用固定 180° 绕 x：等价于 y,z 取反，比齐次坐标乘法快
                V180 = V.astype(np.float32, copy=False)
                V180 = V180 @ _ROT180_X.T   # 或者 V180[:,1:] *= -1
                tm = trimesh.Trimesh(vertices=V180, faces=faces, process=False)
                m = pyrender.Mesh.from_trimesh(tm, material=self.material, smooth=smooth)
                n = self.scene.add(m, name='mesh')
                nodes.append(n)
            rgb, depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
            return rgb[:, :, :3].astype(np.float32), depth
        finally:
            # 清理本帧加入的 mesh nodes
            for n in nodes:
                self.scene.remove_node(n)

    def close(self):
        self.renderer.delete()

# 缓存不同分辨率的 session（同进程内）
@lru_cache(maxsize=4)
def _get_session(width, height):
    return _RendererSession(width, height)

# --- replacement for your function ---
def render_mesh_fast(height, width, meshes, face, cam_param):
    """
    meshes: List[np.ndarray(V,3)], face: (F,3)
    cam_param: {'focal': (fx,fy), 'princpt': (cx,cy)}
    """
    session = _get_session(width, height)                         # 复用 renderer & scene
    session.set_camera(cam_param['focal'], cam_param['princpt'])  # 只更新相机
    # smooth=False 更快，且你的原逻辑也是 False
    rgb, depth = session.render_meshes(meshes, face, smooth=True)
    return rgb, depth

def _Ry(deg):
    rad = np.deg2rad(float(deg))
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[ c, 0,  s],
                     [ 0, 1,  0],
                     [-s, 0,  c]], dtype=np.float32)

def _pre_rotate_for_side_view(V, yaw_deg=90.0):
    """
    给定原始顶点 V（尚未被 render_meshes 里的 Rx180 处理），
    先绕原始质心做 R_pre = Rx180 * Ry(yaw) * Rx180，
    这样在 render_meshes 内部再乘一次 Rx180 后，等价于
    (Rx180 之后) 绕质心做 Ry(yaw) 的侧视旋转。
    """
    center0 = V.mean(axis=0, keepdims=True).astype(np.float32)
    Rx = _ROT180_X  # diag(1,-1,-1)
    Ry = _Ry(yaw_deg)
    R_pre = Rx @ Ry @ Rx  # 注意乘法顺序
    return (V - center0) @ R_pre.T + center0

def render_mesh_fast_side_view(height, width, meshes, face, cam_param, yaw_deg=90.0):
    """
    与 render_mesh_fast 相同的 I/O：渲染“侧视图”（默认 +90°：从 +X 方向看）。
    返回: rgb(float32, HxWx3), depth(float32, HxW)
    """
    # 预旋转：确保进入 session.render_meshes 之前已做 R_pre
    meshes_rot = []
    for V in meshes:
        V = np.asarray(V, dtype=np.float32)
        Vp = _pre_rotate_for_side_view(V, yaw_deg=yaw_deg)  # 改成 -90.0 即左侧
        meshes_rot.append(Vp)

    # 与原版一致：复用 session，设置相机内参，然后渲染
    session = _get_session(width, height)
    session.set_camera(cam_param['focal'], cam_param['princpt'])
    rgb, depth = session.render_meshes(meshes_rot, face, smooth=True)
    return rgb, depth


def render_side_mesh(height, width, meshes, face, cam_param):
    # renderer
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    # scene_bg_color=(1, 1, 1)
    # scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                            #    ambient_light=(0.3, 0.3, 0.3))
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
   
    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # camera
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # mesh
    for mesh in meshes:
        mesh = trimesh.Trimesh(mesh, face)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        # if side_view:
        rot = trimesh.transformations.rotation_matrix(
            np.radians(90), [0, 1, 0])
        # mesh.apply_transform(rot)

        # Manually apply the rotation to each vertex
        vertices_homogeneous = np.hstack((mesh.vertices, np.ones((mesh.vertices.shape[0], 1))))
        transformed_vertices = (vertices_homogeneous @ rot.T)[:, :3]
        mesh.vertices = transformed_vertices

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
        # mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene.add(mesh, 'mesh')

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    renderer.delete()
    return rgb, depth

def _R(axis, deg):
    return trimesh.transformations.rotation_matrix(np.radians(deg), axis)

def render_side_mesh_(height, width, meshes, face, cam_param, yaw_deg=90.0):
    # --- Scene & renderer ---
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.3, 0.3, 0.3))
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE',
                                                  baseColorFactor=(1.0, 1.0, 0.9, 1.0))

    # --- Lights ---
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    for p in ([0, -1, 1], [0, 1, 1], [1, 1, 2]):
        lp = np.eye(4); lp[:3, 3] = np.array(p)
        scene.add(light, pose=lp)

    # --- Camera with intrinsics, BUT rotate camera around Y for side view ---
    focal, princpt = cam_param['focal'], cam_param['princpt']
    cam = pyrender.IntrinsicsCamera(fx=float(focal[0]), fy=float(focal[1]),
                                    cx=float(princpt[0]), cy=float(princpt[1]))
    cam_pose = _R([0, 1, 0], yaw_deg)      # 相机外参：绕 Y 轴旋转 ±90°
    scene.add(cam, pose=cam_pose)

    # --- Mesh nodes: 同样只做 180° X 轴旋转（不要再二次改顶点）---
    model_pose = _R([1, 0, 0], 180.0)
    for V in meshes:
        tri = trimesh.Trimesh(V, face, process=False)
        node = pyrender.Mesh.from_trimesh(tri, material=material, smooth=False)
        scene.add(node, pose=model_pose)

    # --- Render ---
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:, :, :3].astype(np.float32)
    renderer.delete()
    return rgb, depth

union_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'neck',
            9: 'nose',
            10: 'head',
            11: 'lsho',
            12: 'lelb',
            13: 'lwri',
            14: 'rsho',
            15: 'relb',
            16: 'rwri'
        }
def get_flip_paris():
        """
        Get flip pair indices in union order.
        """
        # the same names in union and actual
        flip_pair_names = [['rank', 'lank'], ['rkne', 'lkne'], ['rhip', 'lhip'],
            ['rwri', 'lwri'], ['relb', 'lelb'], ['rsho', 'lsho']]
        union_keys = list(union_joints.keys())
        union_values = list(union_joints.values())

        flip_pairs = [[union_keys[union_values.index(name)] for name in pair] for pair in flip_pair_names]
        return flip_pairs
def get_parent(children):
    n=len(children)
    parent=np.arange(0,n,1,np.uint8)
    for i in range(n):
        for j in children[i]:
            parent[j]=i
    return parent
def fliplr_joints(joints, width, matched_parts, joints_vis=None, is_2d=True):
    """
    flip coords: 2d or 3d joints
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        if is_2d:
            joints_vis[pair[0], :], joints_vis[pair[1], :] = \
                joints_vis[pair[1], :], joints_vis[pair[0], :].copy()
    if is_2d:
        return joints * joints_vis, joints_vis

    return joints

def draw3Dpose(joints_3d,ax,num_joints,cl=200,gt=False):  # blue, orange
    parent_ids = get_parents(num_joints)
    joints_vis = np.ones((joints_3d.shape[0],3))
    X = joints_3d[:, 0]
    Y = joints_3d[:, 1]
    Z = joints_3d[:, 2]
    vis_X = joints_vis[:, 0]
    l = 1.5
    s=1.2
    if gt == 0: #gt
        c = 'g'
    elif gt == 1: #score
        c = 'b'
    elif gt == 2: #min mpjpe
        c = 'y'
    else:
        l = 1
        c = 'c'
        s = 1
    for i in range(0, joints_3d.shape[0]):
        if vis_X[i]:
            ax.scatter(X[i], Y[i], Z[i], c='g',s=s, marker='o')
        x = np.array([X[i], X[parent_ids[i]]], dtype=np.float32)
        y = np.array([Y[i], Y[parent_ids[i]]], dtype=np.float32)
        z = np.array([Z[i], Z[parent_ids[i]]], dtype=np.float32)
        
        ax.plot(x, y, z, c=c,linewidth=l)
    #ax.set_ylim3d(Y[0]-cl,Y[0]+cl)
    #ax.set_zlim3d(Z[0]-cl,Z[0]+cl)
    
    ax.set_title('3d pose')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()

def drawpose(data,num_joints,gt=0):
    parent_ids = get_parents(num_joints)
    for i in range(len(data)):
        color=cmid
        if joints_left.__contains__(i):
                color=cleft
        if joints_right.__contains__(i):
                color=cright
        if gt == 0: #gt
            color = 'g'
        elif gt == 1: #score
            color = 'b'
        elif gt == 2: #min mpjpe
            color = 'y'
        if parent_ids[i]!=i:
            plt.plot([data[i][0],data[parent_ids[i]][0]],[data[i][1],data[parent_ids[i]][1]],color=color,linewidth=1.5)
        #plt.scatter(data[i][0],data[i][1],3,color,4)

def draw(path,data,num_joints,mode=0):
    img = plt.imread(path)
    plt.imshow(img)
    if mode == 0:
        drawpose(data,num_joints=num_joints)
    else:
        for i in range(data.shape[0]):
            plt.scatter(data[i][0],data[i][1],s=4,c='r')
    plt.axis('off')
    plt.show()

def drawbbox(bbox):
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x1 = [bbox[0],bbox[1]]
    x2 = [bbox[0],bbox[3]]
    x3 = [bbox[2],bbox[1]]
    x4 = [bbox[2],bbox[3]]
    plt.plot([x1[0],x2[0]],[x1[1],x2[1]])
    plt.plot([x3[0],x4[0]],[x3[1],x4[1]])
    plt.plot([x1[0],x3[0]],[x1[1],x3[1]])
    plt.plot([x2[0],x4[0]],[x2[1],x4[1]])
    plt.scatter((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2)

def draw3D(data,num_joints):
    fig = plt.figure()
    ax=plt.axes(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(215,270)
    for i in range(data.shape[0]):
        if i==0:
            gt = True
        else:
            gt = False
        draw3Dpose(data[i],ax=ax,num_joints=num_joints,gt=gt)
    plt.gca().set_box_aspect((2, 3.5, 2))
    #ax.set_ylim3d(-400,+400)
    #ax.set_zlim3d(-400,+400)
    #ax.set_xlim3d(-400,+400)
    plt.show()
    plt.close() 

def draw2(path):
    img = plt.imread(path)
    plt.imshow(img)
    plt.show()

def get_parents(num_joints):
    if num_joints==17:
        #print(get_parent(children))
        return get_parent(children)
    else:
        parents = np.zeros(num_joints)
        parents = np.int64(parents)
        parent_ids= db['kintree_table'][0].copy()
        parent_ids[0] = 0
        if num_joints == 24:
            #print(parent_ids)
            return parent_ids
        else:
            parents[:24] = parent_ids
            parents[24] = 15
            parents[25] = 22
            parents[26] = 23
            parents[27] = 10
            parents[28] = 11
            #print(parents)
            return parents
