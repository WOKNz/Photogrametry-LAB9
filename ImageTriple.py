import numpy as np
from ImagePair import ImagePair
from SingleImage import SingleImage
from Camera import Camera
import PhotoViewer as photo
import matplotlib.pyplot as plt
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix, ComputeSkewMatrixFromVector

class ImageTriple(object):
    def __init__(self, imagePair1, imagePair2):
        """
        Inisialize the ImageTriple class

        :param imagePair1: first image pair
        :param imagePair2: second image pair

        .. warning::

            Check if the relative orientation is solved for each image pair
        """
        if not imagePair1.isSolved or not imagePair2.isSolved:
            print('One of the imagePairs Orientation is not solved!')
            return
        self.__imagePair1 = imagePair1
        self.__imagePair2 = imagePair2
        self.__scale = None

    def ComputeScaleBetweenModels(self, cameraPoint1, cameraPoint2, cameraPoint3):
        """
        Compute scale between two models given the relative orientation

        :param cameraPoints1: camera point in first camera space
        :param cameraPoints2: camera point in second camera space
        :param cameraPoints3:  camera point in third camera space
        :return scale: the scale between two models

        :type cameraPoints1: np.array 1x3
        :type cameraPoints2: np.array 1x3
        :type cameraPoints3: np.array 1x3
        :rtype: float
        """
        def vec2mat(v):
            return np.array([[0,-v[2,0],v[1,0]],[v[2,0],0,-v[0,0]],[-v[1,0],v[0,0],0]])
        R1 = np.diag([1,1,1])
        R2 = self.__imagePair1.RotationMatrix_Image2
        R3 = np.dot(R2,self.__imagePair2.RotationMatrix_Image2)
        o1 = np.array([[0,0,0]])
        o2 = self.__imagePair1.PerspectiveCenter_Image2
        # print('o2:\n',o2)
        o3 = o2 + np.dot(R2, self.__imagePair2.PerspectiveCenter_Image2)
        # print('o3:\n',o3)
        v1 = cameraPoint1.reshape(3,1)
        v2 = np.dot(R2,cameraPoint2.reshape(3,1))
        v3 = np.dot(R3,cameraPoint3.reshape(3,1))
                # v1 = cameraPoint1.reshape(3,1) #testing question from lecture
                # v2 = cameraPoint2.reshape(3,1)
                # v3 = cameraPoint3.reshape(3,1)
        d1 = np.dot(vec2mat(v1),v2)
        d2 = np.dot(vec2mat(v2),v3)
        a1 = np.hstack((v1,d1,-v2))
        a2 = np.hstack((o3.reshape(3,1)-o2.reshape(3,1),v3,-d2))
        x1 = np.dot(np.linalg.inv(a1),o2)
        x2 = np.dot(np.linalg.inv(a2),x1[2]*v2)
        self.__scale = x2[0]
        return x2[0]

    def RayIntersection(self, cameraPoints1, cameraPoints2, cameraPoints3):
        """
        Compute coordinates of the corresponding model point

        :param cameraPoints1: points in camera1 coordinate system
        :param cameraPoints2: points in camera2 coordinate system
        :param cameraPoints3: points in camera3 coordinate system

        :type cameraPoints1 np.array nx3
        :type cameraPoints2: np.array nx3
        :type cameraPoints3: np.array nx3

        :return: point in model coordinate system
        :rtype: np.array nx3
        """
        result_Gpoints = []
        dist_e = []

        Ra = Compute3DRotationMatrix(self.__imagePair1.relativeOrientationImage2[3], \
                                     self.__imagePair1.relativeOrientationImage2[4], \
                                     self.__imagePair1.relativeOrientationImage2[5])

        Rb = Compute3DRotationMatrix(self.__imagePair2.relativeOrientationImage2[3], \
                                     self.__imagePair2.relativeOrientationImage2[4], \
                                     self.__imagePair2.relativeOrientationImage2[5])

        R3 = np.dot(Ra, Rb)
        o1 = np.array([[0],[0],[0]])
        o2 = np.array([[self.__imagePair1.relativeOrientationImage2[0]], \
                       [self.__imagePair1.relativeOrientationImage2[1]], \
                       [self.__imagePair1.relativeOrientationImage2[2]]])
        b23 = np.array([[self.__imagePair2.relativeOrientationImage2[0]], \
                        [self.__imagePair2.relativeOrientationImage2[1]], \
                        [self.__imagePair2.relativeOrientationImage2[2]]])
        o3 = o2 + self.__scale[0] * np.dot(R3, b23)


        for i in range(cameraPoints1.shape[0]):  # calculating per point set
            # following the geometric method for forward intersection:
            x_img1 = cameraPoints1[i, :] / 1000  # to meter
            x_img2 = cameraPoints2[i, :] / 1000
            x_img3 = cameraPoints3[i, :] / 1000


            v_img1 = (x_img1).reshape(3, 1)
            v_img2 = (np.dot(Ra, x_img2)).reshape(3, 1)
            v_img3 = (np.dot(R3, x_img3)).reshape(3, 1)
            v_img1 /= np.linalg.norm(v_img1)  # normalization
            v_img2 /= np.linalg.norm(v_img2)
            v_img3 /= np.linalg.norm(v_img3)

            # Creating proper vectors
            vvt_img1 = np.dot(v_img1, v_img1.T)
            vvt_img2 = np.dot(v_img2, v_img2.T)
            vvt_img2 = np.dot(v_img3, v_img3.T)
            I = np.eye(v_img1.shape[0])

            # Partial derivatives
            A_img1 = I - v_img1
            A_img2 = I - v_img2
            A_img3 = I - v_img3


            # L vector
            l1 = np.dot(A_img1, o1)
            l2 = np.dot(A_img2, o2)
            l3 = np.dot(A_img3, o3)

            # Stack
            A = np.vstack((A_img1, A_img2, A_img3))
            l = np.vstack((l1, l2, l3))

            # Direct solution (no iterations needed)
            X = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, l))

            result_Gpoints.append([X[0,0],X[1,0],X[2,0]])

        return np.array(result_Gpoints)

    def drawModles(self, imagePair1, imagePair2, modelPoints1, modelPoints2):
        """
        Draw two models in the same figure

        :param imagePair1: first image pair
        :param imagePair2:second image pair
        :param modelPoints1: points in the first model
        :param modelPoints2:points in the second model

        :type modelPoints1: np.array nx3
        :type modelPoints2: np.array nx3

        :return: None
        """
        figempty = plt.figure()
        ax2 = figempty.add_subplot(111, projection='3d')
        fig1, ax1 = imagePair1.drawImagePair(imagePair1, modelPoints1,figempty,ax2)
        fig, ax = imagePair2.drawImagePair(imagePair2, modelPoints2,fig1,ax1)

        plt.show()


        print('Debug me please!')


if __name__ == '__main__':
    camera = Camera(152, None, None, None, None)
    image1 = SingleImage(camera)
    image2 = SingleImage(camera)
    image3 = SingleImage(camera)
    imagePair1 = ImagePair(image1, image2)
    imagePair2 = ImagePair(image2, image3)
    imageTriple1 = ImageTriple(imagePair11, imagePair22)
