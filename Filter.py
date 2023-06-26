# ------------------------------IMPORT----------------------------------
import numpy as np
import cv2
# ------------------------------COSTANT---------------------------------
# Kalman filterS configuration parameters
# may vary DT for better result, other parameter should be changed only if knowing exactly what you are doing
DT = 1 
A_X = 0.1
A_Y = 0.1
SD_ACC = 0.1
X_SD = 0.05
Y_SD = 0.05
# ------------------------------CLASSES---------------------------------

# Class that implement the Kalman filter
class KalmanFilter(object):
    
    def __init__(self, dt, a_x, a_y, sd_acceleration, x_sd, y_sd):

        self.dt = dt

        # the acceleration which is essentially u from the state update equation 
        self.a = np.matrix([[a_x],[a_y]])

        #  The state transition matrix 
        self.A = np.matrix([[1, 0, self.dt, 0],[0, 1, 0, self.dt],[0, 0, 1, 0],[0, 0, 0, 1]])

        # The control input transition matrix 
        self.B = np.matrix([[(self.dt**2)/2, 0],[0,(self.dt**2)/2],[self.dt,0],[0,self.dt]])

        # The matrix that maps state vector to measurement 
        self.H = np.matrix([[1, 0, 0, 0],[0, 1, 0, 0]])

        # Processs Covariance that for our case depends solely on the acceleration  
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],[0, (self.dt**4)/4, 0, (self.dt**3)/2], [(self.dt**3)/2, 0, self.dt**2, 0],[0, (self.dt**3)/2, 0, self.dt**2]]) * sd_acceleration**2

        # Measurement Covariance
        self.R = np.matrix([[x_sd**2,0], [0, y_sd**2]])

        # The error covariance matrix that is Identity for now. It gets updated based on Q, A and R.
        self.P = np.eye(self.A.shape[1])
        
        #  Finally the vector in consideration ; it's [ x position ;  y position ; x velocity ; y velocity ; ]
        self.x = np.matrix([[0], [0], [0], [0]])

    def predict(self):
        '''
        # The state update : X_t = A*X_t-1 + B*u 
        # here u is acceleration,a 
        '''
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.a)
        
        # Updation of the error covariance matrix 
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        return self.x[0:2]

    def update(self, z):
        ''''update function '''
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) 
    
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))  

        I = np.eye(self.H.shape[1])

        self.P = (I -(K*self.H))*self.P  
        
        return self.x[0:2]

# Class for keeping track of detected keypoint (position, label, other relevant stuff...)
class TrackedKeypoint(object):

    def __init__(self, measured_coords, label, prediction_count, prediction_flag):
        # self.prediction = np.asarray(detected_coords)
        self.measured_coords = measured_coords
        self.label = label
        self.prediction_count = prediction_count
        self.prediction_flag = prediction_flag
        self.filter = KalmanFilter(DT, A_X, A_Y, SD_ACC, X_SD, Y_SD)        
        self.prediction_story = []

# Class wrapper that runs kalman filter for every tracked keypoint
class KalmanWrapper():

    def __init__(self, max_prediction):
        self.max_prediction = max_prediction
        self.tracked_keypoints = {}

    def update(self, measurement):
        '''update all tracked keypoint'''

        # init tracked keypoints list, should be executed only once
        if len(self.tracked_keypoints) == 0:
            for key in measurement.keys():
                self.tracked_keypoints.update( { key: TrackedKeypoint(measurement[key], key, 0, False) } )

        # check if all previously tracked keypoint are found, otherwise increment keypoint's prediction_count value
        for key in self.tracked_keypoints.keys():
            # if tracked keypoint key not found in current measurement dictionary, then increment prediction_count
            if key not in measurement:
                # print('keypoint {} not found, flagged for prediction'.format(key))
                self.tracked_keypoints[key].prediction_flag = True

        # check if some tracked keypoint have been predicted a bit too much
        deleted_keypoint = []
        for tracked_keypoint in self.tracked_keypoints.values():
            # if tracked keypoint prediction count greater than max possible prediction, then remove tracked keypoint
            if tracked_keypoint.prediction_count > self.max_prediction:
                # print('too many predictions for keypoint {}'.format(key))
                deleted_keypoint.append(tracked_keypoint.label)
        for i in range(len(deleted_keypoint)):
            del self.tracked_keypoints[deleted_keypoint[i]]

        # check if there are some newly tracked keypoint and add it to the list
        for key in measurement.keys():
            # if keypoint not found, then add it to tracked keypoint list
            if key not in self.tracked_keypoints:
                # print('keypoint {} newly founded, adding to list'.format(key))
                self.tracked_keypoints.update( { key: TrackedKeypoint(measurement[key], key, 0, False)} )

        # perform predict and/or update for all tracked keypoint
        filtered_keypoint_dict = {}
        for tracked_keypoint in self.tracked_keypoints.values():
            coords: any
            # Check if keypoint can be updated
            if tracked_keypoint.prediction_flag == False:
                # keypoint found, resetting prediction counter
                tracked_keypoint.prediction_count = 0
                # get newly measured keypoint coords
                coords = measurement.get(tracked_keypoint.label)
            else:
                # incrementing prediction counter
                tracked_keypoint.prediction_count += 1
                # predict new coords
                coords = tracked_keypoint.filter.predict()
            # update coords with newly measured coords
            tracked_keypoint.measured_coords = tracked_keypoint.filter.update(coords)
            # print("updated coordinate={} for keypoint={}".format(tracked_keypoint.measured_coords, tracked_keypoint.label))
            # adding coords to coords_story list
            tracked_keypoint.prediction_story.append(coords)
            # adding coords to dictionary
            filtered_keypoint_dict.update({ tracked_keypoint.label: coords})
        # return keypoint dictionary
        return filtered_keypoint_dict
            
# -----------------------------FUNCTION---------------------------------

# Util function for visual keypoint stability feedback
def draw_keypoint_on_image(image, keypoint_dict):
    image = np.zeros((192, 192, 3), dtype=np.uint8)
    for key in keypoint_dict.keys():
        centro = (int(keypoint_dict[key][0]), int(keypoint_dict[key][1]))
        image = cv2.circle(image, centro, 1, (255, 0, 0), 2)
    return image
