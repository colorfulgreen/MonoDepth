import torch
import numpy as np


class BackprojectDepth(object):
    '''Transform a depth image (in pixel frame) into a point cloud (in camera frame). '''

    def __init__(self, width, height, device, batch_size):
        self.batch_size = batch_size
        self.width = width
        self.height = height
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        meshgrid = torch.from_numpy(np.stack(meshgrid, axis=0)).to(device)
        self.pixel_coords = torch.unsqueeze(torch.stack([meshgrid[0].view(-1),
                                                         meshgrid[1].view(-1)], 0), 0)
        self.pixel_coords = self.pixel_coords.repeat(self.batch_size, 1, 1)
        self.ones = torch.ones(self.batch_size, 1, self.height * self.width).to(device)
        self.pixel_coords = torch.cat([self.pixel_coords,
                                       self.ones], dim=1)

    def __call__(self, depth, inv_K):
        '''Returns:
            cam_points: 像素对应的相机坐标
        '''
        cam_points = torch.matmul(inv_K, self.pixel_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points


class Project3D(object):
    '''Project 3D points into a camera with intrinsics K and at position T'''

    def __init__(self, width, height, batch_size):
        self.batch_size = batch_size
        self.width = width
        self.height = height

    def __call__(self, points, K, T):
        '''
        Args:
            T(4x4): rigid transformation written in homogenous coordinates
                    [R   t]
                    [0^T 1]
        '''
        P = torch.matmul(K, T)
        pixel_points = torch.matmul(P, points)

        pixel_coords = pixel_points[:, :2, :] / (pixel_points[:, 2, :].unsqueeze(1) + 1e-7)  # z=1
        pixel_coords = pixel_coords.view(self.batch_size, 2, self.height, self.width)
        pixel_coords = pixel_coords.permute(0, 2, 3, 1)

        pixel_coords[..., 0] /= self.width - 1  # normalized, 0 if on extreme left, 1 if on extreme right
        pixel_coords[..., 1] /= self.height - 1
        pixel_coords = (pixel_coords - 0.5) * 2 # [-1, 1]
        return pixel_coords

# utils
def euler2mat(angle):
    # TODO
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


# utils
def pose_vec2mat(vec):
    '''Convert 6DoF parameters to transformation matrix (rotation_mode=euler)

    Args:
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        transform_mat: A transformation matrix -- [B, 3, 4]
    '''
    translation = vec[:, :3].unsqueeze(-1)
    rot = vec[:, 3:]
    rot_mat = euler2mat(rot)
    transform_mat = torch.cat([rot_mat, translation], dim=2)
    return transform_mat


if __name__ == '__main__':
    # proj = BackprojectDepth(4, 8)
    # print(proj.pixel_coords)

    vec = torch.Tensor([[ 3.7030e-04,  9.4047e-05, -4.7031e-04,  3.0031e-04, -3.6186e-04, 7.2899e-04]])
    transform_mat = pose_vec2mat(vec)
    print(transform_mat)
