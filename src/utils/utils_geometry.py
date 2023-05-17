import torch 

def eulerToMatrix(r_eu:torch.Tensor):
    R_x = torch.tensor([
        [torch.cos(r_eu[0]), torch.sin(r_eu[0]), 0],
        [-torch.sin(r_eu[0]), torch.cos(r_eu[0]), 0],
        [ 0, 0, 1]
    ])
    R_y = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(r_eu[1]), -torch.sin(r_eu[1])],
        [0, torch.sin(r_eu[1]), torch.cos(r_eu[1])]
    ])
    R_x = torch.tensor([
        [torch.cos(r_eu[2]), torch.sin(r_eu[2]), 0],
        [-torch.sin(r_eu[2]), torch.cos(r_eu[2]), 0],
        [ 0, 0, 1]
    ])
    return R_x@R_y@R_x

def angleAndAxisToMatrix(u, theta):
    num_samples = theta.shape[0]
    cross_matrix = torch.zeros((num_samples,3,3))
    cross_matrix[:,0,1] = -u[:,2]
    cross_matrix[:,0,2] = u[:,1]
    cross_matrix[:,1,2] = -u[:,0]
    cross_matrix[:,1,0] = u[:,2]
    cross_matrix[:,2,:] = -cross_matrix[:,:,2]

    R = torch.cos(theta)[...,None,None]*torch.eye(3)[None,...].repeat(num_samples,1,1) + torch.sin(theta)[...,None,None]*cross_matrix + (1-torch.cos(theta))[...,None,None]*torch.einsum("bi,bj->bij",u,u)
    return R

def unproject_rays(K, h_rgb, w_rgb):
    umap = torch.linspace(0.5, w_rgb-0.5, w_rgb)
    vmap = torch.linspace(0.5, h_rgb-0.5, h_rgb)
    umap, vmap = torch.meshgrid(umap, vmap, indexing='xy')
    points_2d = torch.stack((umap, vmap, torch.ones_like(umap)), -1)
    
    # Rays to concatenate with RGB image are unprojected with the same shape
    local_rays = torch.einsum("ij,mnj -> mni",torch.linalg.inv(K[:3,:3]),points_2d)
    local_rays = torch.cat((local_rays,torch.ones(local_rays.shape[:-1])[...,None]),axis=-1)

    return local_rays
################################
# Virtual cameras generation
###############################
def pointcloudCentroid(points_3d_camera, mask_depth, e_v, e_c):
    assert mask_depth.sum()>0, "Depth threshold is to high and there is no valid depth to unproject!"
    weights = mask_depth*1.0                        
    weights /= weights.sum((1,2)).reshape(-1,1,1)

    c_i = ((points_3d_camera[...,:3] +e_v[:,None,None,:])*weights.unsqueeze(-1)).sum((1,2)) + e_c
    return c_i

def rotateZtoCentroid(R, c_i):
    z = torch.tensor([[0,0,1.0]])
    u = torch.cross(z/z.norm(dim=-1)[...,None],c_i/c_i.norm(dim=-1)[...,None], dim=-1)
    theta = torch.asin(u.norm(dim=-1))
    return R @ angleAndAxisToMatrix(u/u.norm(dim=-1)[...,None], theta)

def generateVirtualCameras(input_T_world_camera, points_3d_camera, mask_depth, e_v, e_c):
    T_virtual = input_T_world_camera.clone()
    T_virtual[:,:3,3] += e_v 
    
    c_i = pointcloudCentroid(points_3d_camera, mask_depth, e_v, e_c)
    T_virtual[:,:3,:3] = rotateZtoCentroid(T_virtual[:,:3,:3], c_i)

    return T_virtual