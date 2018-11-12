import numpy as np
import cv2
import glob
import pickle
import os

# prepare object points
n_cols = 9
n_rows = 6
objp = np.zeros((n_cols*n_rows,3), np.float32)
objp[:,:2] = np.mgrid[0:n_cols,0:n_rows].T.reshape(-1,2)

objpoints = []
imgpoints = []

calibration_images_path='camera_cal/calibration*.jpg'
images = glob.glob(calibration_images_path)


for idx, filename in enumerate(images):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (n_cols, n_rows), None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

        cv2.drawChessboardCorners(img, (n_cols, n_rows), corners, ret)
        write_name = 'corners_found'+str(idx)+'.jpg'
        path = './camera_cal/'
        cv2.imwrite(os.path.join(path , write_name), img)
    else:
        print('Unable to calibrate', filename)

img = cv2.imread('./camera_cal/calibration2.jpg')
img_size = (img.shape[1], img.shape[0])

def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(objpoints, imgpoints)
    return cv2.undistort(img, mtx, dist, None, mtx)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
write_name = 'corners_found4_undistorted.jpg'
path = './camera_cal/'
cv2.imwrite(os.path.join(path , write_name), cal_undistort(cv2.imread('./camera_cal/corners_found4.jpg'), objpoints, imgpoints))

dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "./camera_cal/calibration_pickle.p", "wb") )
