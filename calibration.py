import numpy as np
import cv2 as cv
import glob

# followed this tutortial: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
 
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
images = glob.glob('calibration/*.jpg')
 
for fname in images:
    img = cv.imread(fname)
    # cv.imshow('image',img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        # cv.imshow('img', img)
        cv.waitKey(500)

# calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print('camera matrix: ', mtx)
print('distortion coefficients: ', dist)
print('rotation vectors: ', rvecs)
print('translation vectors: ', tvecs)

with open('calibration_data.txt', 'w') as f:
    f.write('camera matrix:\n')
    f.write(np.array2string(mtx, separator=', '))
    f.write('\n\ndistortion coefficients:\n')
    f.write(np.array2string(dist, separator=', '))
    f.write('\n\nrotation vectors:\n')
    for rvec in rvecs:
        f.write(np.array2string(rvec, separator=', '))
        f.write('\n')
    f.write('\ntranslation vectors:\n')
    for tvec in tvecs:
        f.write(np.array2string(tvec, separator=', '))
        f.write('\n')


# undistortion
img = cv.imread('calibration/img18.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
cv.imwrite('undistorted_img18.jpg', dst)

cv.imshow('undistorted', dst)
cv.waitKey(10000)

cv.destroyAllWindows()