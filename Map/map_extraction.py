import numpy as np
import open3d as o3d
import pandas as pd


# Load saved point cloud and visualize it
pcd_load = o3d.io.read_point_cloud("map_palo_alto.pcd")


# convert Open3D.o3d.geometry.PointCloud to numpy array
xyz_load = np.asarray(pcd_load.points)

data = pd.DataFrame(pcd_load.points,columns=['x', 'y', 'z'])
print("x range: ", min(data['x']), max(data['x']))
print("y range: ", min(data['y']), max(data['y']))
print("z range: ", min(data['z']), max(data['z']))

test = data.loc[data['z'] <= 40].to_numpy()


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(test)

o3d.visualization.draw_geometries([pcd])




