import numpy as np

def extrinsic_matrix(look,up,cam):
    """
    builds extrinsic transformation matrix: world_3D-> cam_3D
    """

    z_c = (look - cam)
    z_c = z_c/np.linalg.norm(z_c,keepdims=True)
    x_c = np.cross(up,z_c)
    x_c = x_c/np.linalg.norm(x_c,keepdims=True)
    y_c = np.cross(z_c,x_c)

    M = np.eye(4,4)
    M[:,:3] = np.array([x_c,y_c,z_c,cam]).T
    return M

def generate_rays(fov,H,W):
    x_ndc = np.linspace(-1,1,num=W,endpoint=False)[np.newaxis,...].repeat(H,axis=0) + 1/W
    y_ndc = np.linspace(1,-1,num=H,endpoint=False)[...,np.newaxis].repeat(W,axis=1) - 1/H

    x_cam = x_ndc*np.tan(fov/360*np.pi)*(W/H)
    y_cam = y_ndc*np.tan(fov/360*np.pi)
    z_cam = np.ones((H,W))
    directions = np.stack([[x_cam,y_cam,z_cam,z_cam], -1])
    
    return directions

def rotate_vectors(vs,v1,v2=np.array([0,0,1])):
        """ 
        assume all v1,v2,vs are normalized.
        We rotate the v2 vector to v1, which defines our rotation matrix.
        vs vectors undergo same rotation.
        Imp: Used mostly for rotating cannonical vectors to some orientation,
        hence default v2 is z-axis.
        """
        if len(v2.shape) == 1:
            v2 = v2[np.newaxis,:]
        n = np.cross(v2,v1)
        sine = np.linalg.norm(n,axis=-1)
        cosine = np.sum(v1*v2,axis=-1)
        n = n/np.linalg.norm(n,axis=-1,keepdims=True)
        
        x_plane =np.cross(n,v2)

        vs_z = np.sum(vs*n,axis=-1)
        vs_y = np.sum(vs*v2,axis=-1)
        vs_x = np.sum(vs*x_plane,axis=-1)

        rot_matrix = np.stack([np.stack([cosine,-sine],axis=-1),
                            np.stack([sine,cosine],axis=-1)],
                            axis= -1)
        rot_perp_n = np.einsum('nij,nj->ni', rot_matrix,np.stack([vs_x,vs_y],axis=-1))
        rot_perp_n = rot_perp_n[:,1][:,np.newaxis]*v2 + rot_perp_n[:,0][:,np.newaxis]*x_plane
        rot_vs = rot_perp_n + vs_z[:,np.newaxis]*n

        ### handling 0 rotation case, v1==v2
        rot_vs = np.where((sine==0)[:,np.newaxis],
                        vs*cosine[:,np.newaxis],
                        rot_vs)
        rot_vs = rot_vs/np.linalg.norm(rot_vs,axis=-1,keepdims=True)
        return rot_vs

def reflect_along_normal(dirs,normals):
     
     return dirs - 2*(dirs*normals).sum(axis=-1,keepdims=True)*normals