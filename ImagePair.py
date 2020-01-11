import numpy as np
import numpy.linalg as la

import PhotoViewer as photo
from Camera import Camera
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix, ComputeSkewMatrixFromVector
from SingleImage import SingleImage


# import pandas as pd

class ImagePair(object):

    def __init__(self, image1, image2):
        """
        Initialize the ImagePair class
        :param image1: First image
        :param image2: Second image
        """
        self.__image1 = image1
        self.__image2 = image2
        self.__relativeOrientationImage1 = np.array([0, 0, 0, 0, 0, 0])  # The relative orientation of the first image
        self.__relativeOrientationImage2 = None  # The relative orientation of the second image
        self.__absoluteOrientation = None
        self.__isSolved = False  # Flag for the relative orientation

    @property
    def isSolved(self):
        """
        Flag for the relative orientation
        returns True if the relative orientation is solved, otherwise it returns False

        :return: boolean, True or False values
        """
        return self.__isSolved

    @property
    def relativeOrientationImage1(self):
        """
        Flag for the relative orientation
        returns True if the relative orientation is solved, otherwise it returns False

        :return: boolean, True or False values
        """
        return self.__relativeOrientationImage1

    @property
    def relativeOrientationImage2(self):
        """
        Flag for the relative orientation
        returns True if the relative orientation is solved, otherwise it returns False

        :return: boolean, True or False values
        """
        return self.__relativeOrientationImage2

    @property
    def RotationMatrix_Image1(self):
        """
        return the rotation matrix of the first image

        :return: rotation matrix

        :rtype: np.array 3x3
        """
        return Compute3DRotationMatrix(self.__relativeOrientationImage1[3], self.__relativeOrientationImage1[4],
                                       self.__relativeOrientationImage1[5])

    @property
    def RotationMatrix_Image2(self):
        """
        return the rotation matrix of the second image

        :return: rotation matrix

        :rtype: np.array 3x3
        """
        return Compute3DRotationMatrix(self.__relativeOrientationImage2[3], self.__relativeOrientationImage2[4],
                                       self.__relativeOrientationImage2[5])

    @property
    def PerspectiveCenter_Image1(self):
        """
        return the perspective center of the first image

        :return: perspective center

        :rtype: np.array (3, )
        """
        return self.__relativeOrientationImage1[0:3]

    @property
    def PerspectiveCenter_Image2(self):
        """
        return the perspective center of the second image

        :return: perspective center

        :rtype: np.array (3, )
        """
        return self.__relativeOrientationImage2[0:3]

    def ImagesToGround(self, imagePoints1, imagePoints2, Method=None):
        """
        Computes ground coordinates of homological points

        :param imagePoints1: points in image 1
        :param imagePoints2: corresponding points in image 2
        :param Method: method to use for the ray intersection, three options exist: geometric, vector, Collinearity

        :type imagePoints1: np.array nx2
        :type imagePoints2: np.array nx2
        :type Method: string

        :return: ground points, their accuracies.

        :rtype: dict

        .. warning::

            This function is empty, need implementation


        **Usage example**

        .. code-block:: py

            camera = Camera(152, None, None, None, None)
            image1 = SingleImage(camera)
            image2 = SingleImage(camera)

            imagePoints1 = np.array([[-4.83,7.80],
                                [-4.64, 134.86],
                                [5.39,-100.80],
                                [4.58,55.13],
                                [98.73,9.59],
                                [62.39,128.00],
                                [67.90,143.92],
                                    [56.54,-85.76]])
            imagePoints2 = np.array([[-83.17,6.53],
                                 [-102.32,146.36],
                                 [-62.84,-102.87],
                                 [-97.33,56.40],
                                 [-3.51,14.86],
                                 [-27.44,136.08],
                                 [-23.70,152.90],
                                 [-8.08,-78.07]])

            new = ImagePair(image1, image2)

            new.ImagesToGround(imagePoints1, imagePoints2, 'geometric'))

        """
        picpoints_1_mm = self.__image1.ImageToCamera(imagePoints1)
        picpoints_2_mm = self.__image1.ImageToCamera(imagePoints2)
        exori_XYZ_1 = self.__image1.exteriorOrientationParameters[0:3]
        exori_XYZ_2 = self.__image2.exteriorOrientationParameters[0:3]

        result_Gpoints = []
        dist_e = []

        for i in range(picpoints_1_mm.shape[0]): #calculating per point set
            # following the geometric method for forward intersection:
            x_img1 = np.hstack((picpoints_1_mm[i, :], -self.__image1.camera.focalLength)) / 1000  # to meter
            x_img2 = np.hstack((picpoints_2_mm[i, :], -self.__image2.camera.focalLength)) / 1000
            v_img1 = (np.dot(Compute3DRotationMatrix(self.__image1.exteriorOrientationParameters[3],\
                                                     self.__image1.exteriorOrientationParameters[4],\
                                                     self.__image1.exteriorOrientationParameters[5]), x_img1)).reshape(3, 1)  # Rotating vector +T
            v_img2 = (np.dot(Compute3DRotationMatrix(self.__image1.exteriorOrientationParameters[3],\
                                                     self.__image1.exteriorOrientationParameters[4],\
                                                     self.__image1.exteriorOrientationParameters[5]), x_img2)).reshape(3, 1)  # Rotating vector +T
            v_img1 /= la.norm(v_img1)  # normalization
            v_img2 /= la.norm(v_img2)

            # Creating proper vectors
            vvt_img1 = np.dot(v_img1, v_img1.T)
            vvt_img2 = np.dot(v_img2, v_img2.T)
            I = np.eye(v_img1.shape[0])

            # Partial derivatives
            A_img1 = I - v_img1
            A_img2 = I - v_img2

            # L vector
            l1 = np.dot(A_img1, self.PerspectiveCenter_Image1)
            l2 = np.dot(A_img2, self.PerspectiveCenter_Image2)

            # Stack
            A = np.vstack((A_img1, A_img2))
            l = np.vstack((l1.reshape(3,1), l2.reshape(3,1)))

            # Direct solution (no iterations needed)
            X = np.dot(la.inv(np.dot(A.T, A)), np.dot(A.T, l))
            # dist_e1 = np.dot((I - vvt_img1), X - exori_XYZ_1)
            dist_e1 = np.dot(A_img1, X)- l1.reshape(1,3)
            # dist_e2 = np.dot((I - vvt_img2), X - exori_XYZ_2)
            dist_e2 = np.dot(A_img2, X)- l2.reshape(1,3)

            dist_e.append((np.abs(dist_e1) + np.abs(dist_e2)) / 2) #Average
            result_Gpoints.append([X[0,0],X[1,0],X[2,0]])
        return np.array(result_Gpoints), np.array(dist_e)

    def ComputeDependentRelativeOrientation(self, imagePoints1, imagePoints2, initialValues):
        """
         Compute relative orientation parameters

        :param imagePoints1: points in the first image [m"m]
        :param imagePoints2: corresponding points in image 2(homology points) nx2 [m"m]
        :param initialValues: approximate values of relative orientation parameters

        :type imagePoints1: np.array nx2
        :type imagePoints2: np.array nx2
        :type initialValues: np.array (1x5)

        :return: relative orientation parameters.

        :rtype: np.array 1x6

        **Usage example**

        .. code-block:: py

            camera = Camera(152, None, None, None, None)
            image1 = SingleImage(camera)
            image2 = SingleImage(camera)

            imagePoints1 = np.array([[-4.83,7.80],
                                [-4.64, 134.86],
                                [5.39,-100.80],
                                [4.58,55.13],
                                [98.73,9.59],
                                [62.39,128.00],
                                [67.90,143.92],
                                [56.54,-85.76]])
            imagePoints2 = np.array([[-83.17,6.53],
                                 [-102.32,146.36],
                                 [-62.84,-102.87],
                                 [-97.33,56.40],
                                 [-3.51,14.86],
                                 [-27.44,136.08],
                                 [-23.70,152.90],
                                 [-8.08,-78.07]])
            new = ImagePair(image1, image2)

            new.ComputeDependentRelativeOrientation(imagePoints1, imagePoints2, np.array([1, 0, 0, 0, 0, 0])))

        """
        #Adding Z as -f
        z = (np.ones((1,len(imagePoints1)))*(-self.__image1.camera.focalLength)).T
        imagePoints1 = np.hstack((imagePoints1,z))
        imagePoints2 = np.hstack((imagePoints2,z))
        dx = np.ones((1,5)).reshape((1,5))
        #iterative solution
        while np.linalg.norm(dx) >= 0.0001:
            # Calculatin A,B,W,M,N,u matrices
            A,B,w = self.Build_A_B_W(imagePoints1,imagePoints2,initialValues.reshape((initialValues.size,1)))
            M = np.dot(B, B.T)
            N = np.dot(A.T, np.dot(la.inv(M), A))
            u = np.dot(A.T, np.dot(la.inv(M), w))
            #delta calculating
            dx = -np.dot(la.inv(N), u)
            initialValues += dx
        #adding bx = 1
        initialValues = np.insert(initialValues, 0, 1)
        self.__relativeOrientationImage2 = initialValues
        v = -np.dot(B.T, np.dot(la.inv(M), w))
        sig2 = np.dot(v.T, v) / (A.shape[0] - 5)
        sig_x = np.sqrt(np.diag(sig2 * la.inv(N)))
        #print(pd.DataFrame(np.linalg.inv(N)))
        self.__isSolved = True
        return initialValues, sig_x

    def Build_A_B_W(self, cameraPoints1, cameraPoints2, x):
        """
        Function for computing the A and B matrices and vector w.
        :param cameraPoints1: points in the first camera system
        :param ImagePoints2: corresponding homology points in the second camera system
        :param x: initialValues vector by, bz, omega, phi, kappa ( bx=1)

        :type cameraPoints1: np.array nx3
        :type cameraPoints2: np.array nx3
        :type x: np.array (5,1)

        :return: A ,B matrices, w vector

        :rtype: tuple
        """
        numPnts = cameraPoints1.shape[0]  # Number of points

        dbdy = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        dbdz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

        dXdx = np.array([1, 0, 0])
        dXdy = np.array([0, 1, 0])

        # Compute rotation matrix and it's derivatives
        rotationMatrix2 = Compute3DRotationMatrix(x[2, 0], x[3, 0], x[4, 0])
        dRdOmega = Compute3DRotationDerivativeMatrix(x[2, 0], x[3, 0], x[4, 0], 'omega')
        dRdPhi = Compute3DRotationDerivativeMatrix(x[2, 0], x[3, 0], x[4, 0], 'phi')
        dRdKappa = Compute3DRotationDerivativeMatrix(x[2, 0], x[3, 0], x[4, 0], 'kappa')

        # Create the skew matrix from the vector [bx, by, bz]
        bMatrix = ComputeSkewMatrixFromVector(np.array([1, x[0, 0], x[1, 0]]))

        # Compute A matrix; the coplanar derivatives with respect to the unknowns by, bz, omega, phi, kappa
        A = np.zeros((numPnts, 5))
        A[:, 0] = np.diag(
            np.dot(cameraPoints1,
                   np.dot(dbdy, np.dot(rotationMatrix2, cameraPoints2.T))))  # derivative in respect to by
        A[:, 1] = np.diag(
            np.dot(cameraPoints1,
                   np.dot(dbdz, np.dot(rotationMatrix2, cameraPoints2.T))))  # derivative in respect to bz
        A[:, 2] = np.diag(
            np.dot(cameraPoints1, np.dot(bMatrix, np.dot(dRdOmega, cameraPoints2.T))))  # derivative in respect to omega
        A[:, 3] = np.diag(
            np.dot(cameraPoints1, np.dot(bMatrix, np.dot(dRdPhi, cameraPoints2.T))))  # derivative in respect to phi
        A[:, 4] = np.diag(
            np.dot(cameraPoints1, np.dot(bMatrix, np.dot(dRdKappa, cameraPoints2.T))))  # derivative in respect to kappa

        # Compute B matrix; the coplanar derivatives in respect to the observations, x', y', x'', y''.
        B = np.zeros((numPnts, 4 * numPnts))
        k = 0
        for i in range(numPnts):
            p1vec = cameraPoints1[i, :]
            p2vec = cameraPoints2[i, :]
            B[i, k] = np.dot(dXdx, np.dot(bMatrix, np.dot(rotationMatrix2, p2vec)))
            B[i, k + 1] = np.dot(dXdy, np.dot(bMatrix, np.dot(rotationMatrix2, p2vec)))
            B[i, k + 2] = np.dot(np.dot(p1vec, np.dot(bMatrix, rotationMatrix2)), dXdx)
            B[i, k + 3] = np.dot(np.dot(p1vec, np.dot(bMatrix, rotationMatrix2)), dXdy)
            k += 4

        # w vector
        w = np.diag(np.dot(cameraPoints1, np.dot(bMatrix, np.dot(rotationMatrix2, cameraPoints2.T))))

        return A, B, w

    def ImagesToModel(self, imagePoints1, imagePoints2, Method):
        """
        Mapping points from image space to model space

        :param imagePoints1: points from the first image
        :param imagePoints2: points from the second image
        :param Method: method for intersection

        :type imagePoints1: np.array nx2
        :type imagePoints2: np.array nx2
        :type Method: string

        :return: corresponding model points
        :rtype: np.array nx3


        .. warning::

            This function is empty, need implementation

        .. note::

            One of the images is a reference, orientation of this image must be set.

        """

    def GroundToImage(self, groundPoints):
        """
        Transforming ground points to image points

        :param groundPoints: ground points [m]

        :type groundPoints: np.array nx3

        :return: corresponding Image points

        :rtype: np.array nx2

        """
        pass

        # 1. calculating pic plane
        # 2. calculating intersection of vector from perspective center to ground poind


    def geometricIntersection(self, cameraPoints1, cameraPoints2):
        """
        Ray Intersection based on geometric calculations.

        :param cameraPoints1: points in the first image
        :param cameraPoints2: corresponding points in the second image

        :type cameraPoints1: np.array nx3
        :type cameraPoints2: np.array nx3

        :return: lambda1, lambda2 scalars

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation

        """

    def vectorIntersction(self, cameraPoints1, cameraPoints2):
        """
        Ray Intersection based on vector calculations.

        :param cameraPoints1: points in image space
        :param cameraPoints2: corresponding image points

        :type cameraPoints1: np.array nx
        :type cameraPoints2: np.array nx


        :return: lambda1, lambda2 scalars

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation

        """

    def CollinearityIntersection(self, cameraPoints1, cameraPoints2):
        """
        Ray intersection based on the collinearity principle

        :param cameraPoints1: points in image space
        :param cameraPoints2: corresponding image points

        :type cameraPoints1: np.array nx2
        :type cameraPoints2: np.array nx2

        :return: corresponding ground points

        :rtype: np.array nx3

        .. warning::

            This function is empty, need implementation

        """

    def drawImagePair(self, ImagePair, modelPoints, fig2, ax2):

        """
        Drawing 3D output of given imgPair

        :param ImagePair: ImagePair object
        :param modelPoints: Points In Model

        :type ImagePair: np.array nx2
        :type modelPoints: np.array nx2

        :return: fig,axis
        """
        fig = fig2
        # ax = fig.add_subplot(111, projection='3d')
        ax = ax2
        pix_size = 2.4e-3
        photo.drawOrientation(self.RotationMatrix_Image1,\
                              np.reshape(self.PerspectiveCenter_Image1,\
                                         (self.PerspectiveCenter_Image1.size, 1)),\
                              3,ax)
        photo.drawOrientation(self.RotationMatrix_Image2,\
                           np.reshape(self.PerspectiveCenter_Image2,\
                                      (self.PerspectiveCenter_Image2.size, 1)),\
                              3, ax)

        photo.drawImageFrame(5472 * pix_size, 3648 * pix_size, self.RotationMatrix_Image1,\
                             np.reshape(self.PerspectiveCenter_Image1,\
                                        (self.PerspectiveCenter_Image1.size, 1)),\
                             4248.06 * pix_size, 0.8, ax)
        photo.drawImageFrame(5472 * pix_size, 3648 * pix_size, self.RotationMatrix_Image2,\
                             np.reshape(self.PerspectiveCenter_Image2,\
                                        (self.PerspectiveCenter_Image2.size, 1)),\
                             4248.06 * pix_size, 0.8, ax)

        photo.drawRays(modelPoints*100,\
                       np.reshape(self.PerspectiveCenter_Image1,\
                                  (self.PerspectiveCenter_Image1.size, 1)), ax)
        photo.drawRays(modelPoints*100,\
                    np.reshape(self.PerspectiveCenter_Image2,\
                               (self.PerspectiveCenter_Image2.size, 1)), ax)

        modelPoints = modelPoints*100
        for i in range(0, modelPoints.shape[0]-2, 1):
            ax.plot([modelPoints[i, 0], modelPoints[i + 1, 0]], \
                    [modelPoints[i, 1], modelPoints[i + 1, 1]], \
                    [modelPoints[i, 2], modelPoints[i + 1, 2]], color='blue')


        # Delete in case of diffrent object than this lab
        ax.plot([modelPoints[0, 0], modelPoints[8, 0]], \
                [modelPoints[0, 1], modelPoints[8, 1]], \
                [modelPoints[0, 2], modelPoints[8, 2]], color='blue')
        ax.plot([modelPoints[3, 0], modelPoints[6, 0]], \
                [modelPoints[3, 1], modelPoints[6, 1]], \
                [modelPoints[3, 2], modelPoints[6, 2]], color='blue')
        ax.plot([modelPoints[2, 0], modelPoints[7, 0]], \
                [modelPoints[2, 1], modelPoints[7, 1]], \
                [modelPoints[2, 2], modelPoints[7, 2]], color='blue')

        ax.view_init(20, 80)

        return fig, ax

    def vec2mat(self, v):
        return np.array([[0, -v[2, 0], v[1, 0]], [v[2, 0], 0, -v[0, 0]], [-v[1, 0], v[0, 0], 0]])

    def RotationLevelModel(self, constrain1, constrain2):
        """
        Function that calculates Rotation Matrix
        :param constrain1: First axis vector
        :param constrain2: Second Axis vector
        :return: Rotation Matrix (3X3)
        """
        if la.norm(constrain1[1]) != 1 or la.norm(constrain2[1]) != 1:
            x = (constrain1[1] / la.norm(constrain1[1])).reshape(3, 1)
            z = (constrain2[1] / la.norm(constrain2[1])).reshape(3, 1)

        if constrain1[0] == constrain2[0]:
            return np.diag(np.ones(1, 3))

        y = np.dot(self.vec2mat(z), x) / \
            la.norm(np.dot(self.vec2mat(z), x))

        return np.hstack((x, y, z))

    def ModelTransformation(self, modelPoints, scale, rotationMatrix):
        """
        Calculating rotated and scaled points (I had to add rotation matrix since none of the objects hold world points)
        :param modelPoints: points in model system
        :param scale:  Scale factor from model to ground
        :param rotationMatrix: rotaion matrix from model to ground
        :return: nd.array (nx3)
        """

        return (np.dot(rotationMatrix, modelPoints.T) * scale).T


if __name__ == '__main__':
    camera = Camera(152, None, None, None, None)
    image1 = SingleImage(camera)
    image2 = SingleImage(camera)
    leftCamPnts = np.array([[-4.83, 7.80],
                            [-4.64, 134.86],
                            [5.39, -100.80],
                            [4.58, 55.13],
                            [98.73, 9.59],
                            [62.39, 128.00],
                            [67.90, 143.92],
                            [56.54, -85.76]])
    rightCamPnts = np.array([[-83.17, 6.53],
                             [-102.32, 146.36],
                             [-62.84, -102.87],
                             [-97.33, 56.40],
                             [-3.51, 14.86],
                             [-27.44, 136.08],
                             [-23.70, 152.90],
                             [-8.08, -78.07]])
    new = ImagePair(image1, image2)

    print(new.ComputeDependentRelativeOrientation(leftCamPnts, rightCamPnts, np.array([1, 0, 0, 0, 0, 0])))
