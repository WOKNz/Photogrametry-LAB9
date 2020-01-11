import numpy as np

import MatrixMethods as mt

md1 = np.array([-0.01952,-0.41789,-0.39889]) # Omega Phi Kappa
md2 = np.array([0.00345,-0.33694,-0.31978]) # Omega Phi Kappa model 2
b12 = np.array([[1],[-0.1528],[0.07579]]) # Unscaled airbase b12 model 1
b23 = np.array([[1],[-0.08665],[0.15800]]) # Unscaled airbase b23 model 2

Ra = mt.Compute3DRotationMatrix(md1[0],md1[1],md1[2])
Rb = mt.Compute3DRotationMatrix(md2[0],md2[1],md2[2])
R3 = np.dot(Ra,Rb)

md3 = np.zeros((1,3))
md3[0,0] = np.arctan2(-R3[1,2],R3[2,2])
md3[0,1] = np.arctan2(-R3[0,1],R3[0,0])
md3[0,2] = np.arcsin(R3[0,2])
print('Omega,Phi,Kappa Pic3:\n',md3)



o3 = b12 + np.dot(Ra,b23)

print('o3:\n',o3)

# section 2
# Imports
import numpy as np
from Reader import Reader as rd
import Camera
import SingleImage
from ImagePair import ImagePair
import pandas as pd
import copy
from ImageTriple import ImageTriple

# Setting Parameters
y_size = 3648 # size of y axis of a picture
pix_s = 2.4e-3  #single pixel size (micron should be e-6)
focal = 4239.655*pix_s
ppoint = np.array([2746.295,1837.450])*pix_s
radial_d = np.array([0.0473,-0.414,])*pix_s
dece_d = np.array([-0.0014,-0.0028,0])*pix_s

# Reading and Creating list of Homological points
list_of_raw_points = []
list_of_raw_points.append(rd.ReadSampleFile(r'colorbase/IMG_1793.json'))
list_of_raw_points.append(rd.ReadSampleFile(r'colorbase/IMG_1794.json'))
list_of_raw_points.append(rd.ReadSampleFile(r'colorbase/IMG_1795.json'))

# Fixing camera axis
list_of_y_fixed_points = copy.deepcopy(list_of_raw_points)
for m in range(len(list_of_raw_points)):
    for row in range(len(list_of_raw_points[m]-1)):
        list_of_y_fixed_points[m][row,1] = y_size -  list_of_raw_points[m][row,1]

# Fixing camera center
list_of_y_center_fixed_points = copy.deepcopy(list_of_y_fixed_points)
for m in range(len(list_of_y_fixed_points)):
    for row in range(len(list_of_y_fixed_points[m]-1)):
        list_of_y_center_fixed_points[m][row,:] =\
        list_of_y_fixed_points[m][row,:] - ppoint/pix_s

# Multiply by pixel size
list_of_y_center_fixed_mm_points = copy.deepcopy(list_of_y_center_fixed_points)
for m in range(len(list_of_y_center_fixed_points)):
    for row in range(len(list_of_y_center_fixed_points[m]-1)):
        list_of_y_center_fixed_mm_points[m][row,:] =\
        list_of_y_center_fixed_points[m][row,:]*pix_s

# Creating pairs of point sets
pointset12 = [list_of_raw_points[0],list_of_raw_points[1]]
pointset23 = [list_of_raw_points[1],list_of_raw_points[2]]

# Creating camera objects, we entering raw points of color surface as fudicals
camera1 = Camera.Camera(focal,ppoint,radial_d,dece_d,list_of_y_center_fixed_mm_points[0])
camera2 = Camera.Camera(focal,ppoint,radial_d,dece_d,list_of_y_center_fixed_mm_points[1])
camera3 = Camera.Camera(focal,ppoint,radial_d,dece_d,list_of_y_center_fixed_mm_points[2])

# Creating Image Objects
img1 = SingleImage.SingleImage(camera1)
img2 = SingleImage.SingleImage(camera2)
img3 = SingleImage.SingleImage(camera3)

# Calculating inner orientation

inner_orient_img1 = img1.ComputeInnerOrientation(list_of_raw_points[0])
inner_orient_img2 = img2.ComputeInnerOrientation(list_of_raw_points[1])
inner_orient_img3 = img3.ComputeInnerOrientation(list_of_raw_points[2])

# Creating Image pair objects
img_pair_1 = ImagePair(img1,img2)
img_pair_2 = ImagePair(img2,img3)

# Calculating relative orientation

initial_p = np.array([0.0,0.0,0.0,0.0,0.0])
relative_ori_1,sigma12 = img_pair_1.ComputeDependentRelativeOrientation(list_of_y_center_fixed_mm_points[0],\
                                                                list_of_y_center_fixed_mm_points[1],\
                                                                initial_p)
relative_ori_2,sigma23 = img_pair_2.ComputeDependentRelativeOrientation(list_of_y_center_fixed_mm_points[1],\
                                                                list_of_y_center_fixed_mm_points[2],\
                                                                initial_p)

# Calculating relative model coordinates of object

object_points23 = []
object_points12 = []
object_points23.append(rd.ReadSampleFile(r'colorbase/boxespic2.json'))
object_points23.append(rd.ReadSampleFile(r'colorbase/boxespic3.json'))
object_points12.append(rd.ReadSampleFile(r'colorbase/boxespic1.json'))
object_points12.append(rd.ReadSampleFile(r'colorbase/boxespic2.json'))

# Vectors for Normal
vectors3 = rd.ReadSampleFile((r'3vectors.json'))

world_model_points23 = img_pair_1.ImagesToGround(object_points23[0], \
                                                 object_points23[1])
world_model_points12 = img_pair_1.ImagesToGround(object_points12[0], \
                                                 object_points12[1])

world_model_base12 = img_pair_1.ImagesToGround(list_of_raw_points[0], \
                                               list_of_raw_points[1])
world_model_base23 = img_pair_2.ImagesToGround(list_of_raw_points[1], \
                                               list_of_raw_points[2])


##testing plot of model points
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(0, world_model_base12[0].shape[0]-1, 1):
#     ax.scatter([world_model_base12[0][i, 0]], \
#             [world_model_base12[0][i, 1]], \
#             [world_model_base12[0][i, 2]], color='blue')
# ax.scatter([world_model_base12[0][0, 0]], \
#             [world_model_base12[0][0, 1]], \
#             [world_model_base12[0][0, 2]], color='red')
# ax.scatter([world_model_base12[0][8, 0]], \
#             [world_model_base12[0][8, 1]], \
#             [world_model_base12[0][8, 2]], color='black')
# ax.scatter([world_model_base12[0][3, 0]], \
#             [world_model_base12[0][3, 1]], \
#             [world_model_base12[0][3, 2]], color='green')
# plt.show()
#


## Part of LAB9 - uncomment if needed
#
# dp2p1_camera = np.hstack((list_of_y_center_fixed_mm_points[0][8,:]-list_of_y_center_fixed_mm_points[0][0,:],0))
# dp3p1_camera = np.hstack((list_of_y_center_fixed_mm_points[0][3,:]-list_of_y_center_fixed_mm_points[0][0,:],0))
# fornormvectors = np.vstack((dp2p1_camera,dp3p1_camera))
#
#
def vec2mat(v):
    return np.array([[0, -v[2, 0], v[1, 0]], [v[2, 0], 0, -v[0, 0]], [-v[1, 0], v[0, 0], 0]])


#
# vnormal = np.dot(vec2mat(dp2p1_camera.reshape(3,1)),dp3p1_camera.reshape(3,1))
# print('Vectors for norm Calculating: \n',fornormvectors)
# print('Normal : \n',vnormal)
## End of Part LAB9 - uncomment if needed


img1cameraPoints = img1.ImageToCamera(object_points12[0])
img2cameraPoints = img2.ImageToCamera(object_points12[1])
img3cameraPoints = img3.ImageToCamera(object_points23[1])

# print(pd.DataFrame(world_model_points23[0]))
# print(pd.DataFrame(world_model_points12[0]))

# creating triple Img model

tripleImg = ImageTriple(img_pair_1,img_pair_2)

# Drawing the two models on one plot
# tripleImg.drawModles(img_pair_1,img_pair_2,world_model_points12[0],world_model_points23[0])
#tripleImg.drawModles(img_pair_1,img_pair_2,world_model_points12[0],world_model_points23[0])

# Creating arrays of points by image adding focal length
img1cameraPoints = np.hstack((img1cameraPoints, np.ones((img1cameraPoints.shape[0], 1)) * -focal))
img2cameraPoints = np.hstack((img2cameraPoints, np.ones((img2cameraPoints.shape[0], 1)) * -focal))
img3cameraPoints = np.hstack((img3cameraPoints, np.ones((img3cameraPoints.shape[0], 1)) * -focal))

        # v1 = np.array([50,0,-152])  # testing question from lecture
        # v2 = np.array([23.2885,-8.8863,-151.7961])
        # v3 = np.array([22.8971,-8.5346,-153.6924])
        # scale = tripleImg.ComputeScaleBetweenModels(v1,v2,v3)

# Calculating scale to next model
scale = tripleImg.ComputeScaleBetweenModels(img1cameraPoints[0,:],img2cameraPoints[0,:],img3cameraPoints[0,:])
print('Scale of second model based on point 1:',scale[0])

# Calculating avg of scale based on all points
scales = []
for i in range(img1cameraPoints.shape[0]):
    scales.append(tripleImg.ComputeScaleBetweenModels(img1cameraPoints[i,:],img2cameraPoints[i,:],img3cameraPoints[i,:]))
scale_avg = np.mean(np.array(scales))
scale_var = np.var(np.array(scales))
print('Scale avg of second model based on all points:',scale_avg)
print('Scale var of second model based on all points:',scale_var)

# Calculating new orientation parameters with scale
md1 = np.array([-0.01952,-0.41789,-0.39889]) # Omega Phi Kappa
md2 = np.array([0.00345,-0.33694,-0.31978]) # Omega Phi Kappa model 2
b12 = np.array([[1],[-0.1528],[0.07579]]) # Unscaled airbase b12 model 1
b23 = np.array([[1],[-0.08665],[0.15800]]) # Unscaled airbase b23 model 2

Ra = mt.Compute3DRotationMatrix(md1[0],md1[1],md1[2])
Rb = mt.Compute3DRotationMatrix(md2[0],md2[1],md2[2])
R3 = np.dot(Ra,Rb)

md3 = np.zeros((1,3))
md3[0,0] = np.arctan2(-R3[1,2],R3[2,2])
md3[0,1] = np.arctan2(-R3[0,1],R3[0,0])
md3[0,2] = np.arcsin(R3[0,2])
print('Omega,Phi,Kappa Pic3:\n', md3)

o3 = b12 + scale[0] * np.dot(R3, b23)

print('o3:\n', o3)

# Calculating points from triple model

best_model3_points = tripleImg.RayIntersection(img1cameraPoints, img2cameraPoints, img3cameraPoints)
camera1_mm_base = np.hstack((list_of_y_center_fixed_mm_points[0], \
                             -focal * np.ones((list_of_y_center_fixed_mm_points[0].shape[0], 1))))
camera2_mm_base = np.hstack((list_of_y_center_fixed_mm_points[1], \
                             -focal * np.ones((list_of_y_center_fixed_mm_points[1].shape[0], 1))))
camera3_mm_base = np.hstack((list_of_y_center_fixed_mm_points[2], \
                             -focal * np.ones((list_of_y_center_fixed_mm_points[2].shape[0], 1))))

best_basel3_points = tripleImg.RayIntersection(camera1_mm_base, camera2_mm_base, camera3_mm_base)

print('Triple Model points:\n', best_model3_points)

# LAB9 Starts here!

# Drawing the united model
# tripleImg.drawModles(img_pair_1,img_pair_1,best_model3_points,best_model3_points)

# Q3
# Calculating Rmt 
dp2p1_camera = (best_basel3_points[8, :] - best_basel3_points[0, :]) / \
               np.linalg.norm(best_basel3_points[8, :] - best_basel3_points[0, :])
dp3p1_camera = (best_basel3_points[3, :] - best_basel3_points[0, :]) / \
               np.linalg.norm(best_basel3_points[3, :] - best_basel3_points[0, :])
vnormal = np.dot(vec2mat(dp2p1_camera.reshape(3, 1)), dp3p1_camera.reshape(3, 1))

constrain1 = ('x', dp2p1_camera.reshape(3, 1))
constrain2 = ('z', vnormal)

Rmt = img_pair_1.RotationLevelModel(constrain1, constrain2)

# Calculatin Rtg (using ground measurements)
dp2p1_ground = np.array([34.3, 0, 0]) / np.linalg.norm(np.array([34.3, 0, 0]))
dp3p1_ground = np.array([1.4872, 5.9488, 0]) / np.linalg.norm(np.array([1.4872, 5.9488, 0]))
vnormal = np.dot(vec2mat(dp2p1_ground.reshape(3, 1)), dp3p1_ground.reshape(3, 1))

constrain1_g = ('x', dp2p1_ground)
constrain2_g = ('z', vnormal)

Rgt = img_pair_1.RotationLevelModel(constrain1_g, constrain2_g)

Rmg = np.dot(Rmt, Rgt.T)
(pd.DataFrame(Rmg)).to_excel('Rmg1.xlsx')
print('Rmg:\n', Rmg)
# Q4
# Calculating Rmt
dp2p1_camera = (best_basel3_points[12, :] - best_basel3_points[1, :]) / \
               np.linalg.norm(best_basel3_points[12, :] - best_basel3_points[1, :])
dp3p1_camera = (best_basel3_points[15, :] - best_basel3_points[1, :]) / \
               np.linalg.norm(best_basel3_points[15, :] - best_basel3_points[1, :])

vnormal = np.dot(vec2mat(dp2p1_camera.reshape(3, 1)), dp3p1_camera.reshape(3, 1))

constrain1 = ('x', dp2p1_camera.reshape(3, 1))
constrain2 = ('z', vnormal)

Rmt = img_pair_1.RotationLevelModel(constrain1, constrain2)

# Calculatin Rtg (using ground measurements)
dp2p1_ground = np.array([8 * 1.4872, -7 * 1.4872, 0]) / np.linalg.norm(np.array([8 * 1.4872, -7 * 1.4872, 0]))
dp3p1_ground = np.array([10 * 1.4872, 2 * 1.4872, 0]) / np.linalg.norm(np.array([10 * 1.4872, 2 * 1.4872, 0]))

vnormal = np.dot(vec2mat(dp2p1_ground.reshape(3, 1)), dp3p1_ground.reshape(3, 1))

constrain1_g = ('x', dp2p1_ground.reshape(3, 1))
constrain2_g = ('z', vnormal)

Rgt = img_pair_1.RotationLevelModel(constrain1_g, constrain2_g)

Rmg = np.dot(Rmt, Rgt.T)
(pd.DataFrame(Rmg)).to_excel('Rmg2.xlsx')

print('Rmg:\n', Rmg)

# Distance calculating
# model distance
model3_distances = []
model3_distances.append(np.linalg.norm(best_basel3_points[3, :] - best_basel3_points[0, :]))
model3_distances.append(np.linalg.norm(best_basel3_points[8, :] - best_basel3_points[0, :]))
model3_distances.append(np.linalg.norm(best_basel3_points[3, :] - best_basel3_points[8, :]))
model3_distances.append(np.linalg.norm(best_basel3_points[9, :] - best_basel3_points[4, :]))
model3_distances.append(np.linalg.norm(best_basel3_points[2, :] - best_basel3_points[5, :]))
model3_distances.append(np.linalg.norm(best_basel3_points[13, :] - best_basel3_points[10, :]))
# box elements
model3_distances.append(np.linalg.norm(best_model3_points[0, :] - best_model3_points[1, :]))
model3_distances.append(np.linalg.norm(best_model3_points[3, :] - best_model3_points[4, :]))
model3_distances.append(np.linalg.norm(best_model3_points[4, :] - best_model3_points[0, :]))
model3_distances.append(np.linalg.norm(best_model3_points[4, :] - best_model3_points[2, :]))
model3_distances.append(np.linalg.norm(best_model3_points[3, :] - best_model3_points[2, :]))
# model3_distances.append(np.linalg.norm(best_model3_points[3,:]-best_model3_points[6,:]))
model3_distances.append(np.linalg.norm(best_model3_points[5, :] - best_model3_points[6, :]))
# model3_distances.append(np.linalg.norm(best_model3_points[5,:]-best_model3_points[4,:]))
# model3_distances.append(np.linalg.norm(best_model3_points[8,:]-best_model3_points[0,:]))
# model3_distances.append(np.linalg.norm(best_model3_points[8,:]-best_model3_points[5,:]))
# model3_distances.append(np.linalg.norm(best_model3_points[1,:]-best_model3_points[2,:]))
model3_distances = np.array(model3_distances).reshape(len(model3_distances), 1)

# real distances
l1 = 47.5
l2 = 32.7
l3 = 11.5
real_distances = np.array([37.6, 34.3, 45, 31, 13.2, 33.6, l1, l2, l3, l1, l2, l2]).reshape(len(model3_distances), 1)

# scale calculating

scale_to_real = np.divide(real_distances, model3_distances)
scale_to_real_avg = np.average(scale_to_real)
scale_to_real_var = np.sqrt(np.mean((scale_to_real_avg - scale_to_real) ** 2))
pd.DataFrame(np.hstack((model3_distances, real_distances, scale_to_real))).to_excel('scale_table.xlsx')
print('Scale avg:', scale_to_real_avg)
print('Scale std:', scale_to_real_var)
# Multiplying model by scale
model3_distances_scale439 = model3_distances * 439.7142
model3_distances_scale350 = model3_distances * 350

best_model3_points_scaled = (np.dot(Rmg, best_model3_points.T) * scale_avg).T
# Plot scaled model
tripleImg.drawModles(img_pair_1, img_pair_1, best_model3_points_scaled, best_model3_points)
