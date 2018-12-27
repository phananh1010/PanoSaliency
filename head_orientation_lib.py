import numpy as np
from pyquaternion import Quaternion

H = 90
W = 160

def extract_direction_dataset1(q):
        #q is quaternion
        v0 = [1, 0, 0]
        q = Quaternion([q[3], q[2], q[1], q[0]])
        return q.rotate(v0)
    
def extract_direction_dataset2(q):
        #q is quaternion
        v0 = [0, 0, 1]
        q = Quaternion([q[3], -q[2], q[1], -q[0]])
        return q.rotate(v0)

def pixel_to_ang(_x, _y, _geo_h, _geo_w):
    phi = geoy_to_phi(_x, _geo_h)
    theta = -(_y * 1.0 / _geo_w) * 360
    if theta < -180: theta = 360 + theta
    return theta, phi

def geoy_to_phi(_geoy, _height):
    d = (_height/2 - _geoy) * 1.0 / (_height/2)
    s = -1 if d < 0 else 1
    return s * np.arcsin(np.abs(d)) / np.pi * 180

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def degree_distance(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))/np.pi * 180

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))/np.pi * 180


#lib to create fixation map

def vector_to_ang(_v):
    #v = np.array(vector_ds[0][600][1])
    #v = np.array([0, 0, 1])
    _v = np.array(_v)
    alpha = degree_distance(_v, [0, 1, 0])#degree between v and [0, 1, 0]
    phi = 90.0 - alpha
    proj1 = [0, np.cos(alpha/180.0 * np.pi), 0] #proj1 is the projection of v onto [0, 1, 0] axis
    proj2 = _v - proj1#proj2 is the projection of v onto the plane([1, 0, 0], [0, 0, 1])
    theta = degree_distance(proj2, [1, 0, 0])#theta = degree between project vector to plane and [1, 0, 0]
    sign = -1.0 if degree_distance(_v, [0, 0, -1]) > 90 else 1.0
    theta = sign * theta
    return theta, phi


def ang_to_geoxy(_theta, _phi, _h, _w):
    x = _h/2.0 - (_h/2.0) * np.sin(_phi/180.0 * np.pi)
    temp = _theta
    if temp < 0: temp = 180 + temp + 180
    temp = 360 - temp
    y = (temp * 1.0/360 * _w)
    return int(x), int(y)


