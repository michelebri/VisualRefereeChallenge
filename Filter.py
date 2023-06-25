# ------------------------------IMPORT----------------------------------
import numpy as np
import cv2
# ------------------------------COSTANT---------------------------------
# DT = 0.75
# A_X = 0.1
# A_Y = 0.1
# SD_ACC = 0.1
# X_SD = 0.05
# Y_SD = 0.05
DT = 5
A_X = 2
A_Y = 2
SD_ACC = 1
X_SD = 0.1
Y_SD = 0.1
# ------------------------------CLASSES---------------------------------

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

class TrackedKeypoint(object):

    def __init__(self, measured_coords, label):
        # self.prediction = np.asarray(detected_coords)
        self.measured_coords = measured_coords
        self.label = label
        self.filter = KalmanFilter(DT, A_X, A_Y, SD_ACC, X_SD, Y_SD)        
        self.prediction_story = []

class KalmanWrapper():

    def __init__(self):
        self.tracked_keypoints = []

    def predict(self):
        '''get predictions for all tracked keypoint'''
        # decleare filtered keypoint dictionary
        filtered_keypoint_dict = {}
        for i in range(len(self.tracked_keypoints)):
            # get current tracked keypoint
            keypoint = self.tracked_keypoints[i]
            # predict new coords
            predicted_coords = keypoint.filter.predict()

            # add predicted coords to dictionary
            filtered_keypoint_dict.update({ keypoint.label: predicted_coords})
            # print("predicted coordinate={} for keypoint={}".format(predicted_coords, keypoint.label))

        # return keypoint dictionary
        return filtered_keypoint_dict

    def update(self, measurement):
        '''update all tracked keypoint'''

        # check if there are some tracked keypoint, if not add it from measurement dictionary
        if len(self.tracked_keypoints) == 0:
            for key in measurement.keys():
                self.tracked_keypoints.append( TrackedKeypoint(measurement[key], key) )

        # perform update for all tracked keypoint
        for i in range(len(self.tracked_keypoints)):
            # get current tracked keypoint
            tracked_keypoint = self.tracked_keypoints[i]
            # get newly measured keypoint coords
            updated_coords = tracked_keypoint.measured_coords
            # add newly measured coords to coords_story list
            tracked_keypoint.prediction_story.append(updated_coords)
            # update coords with newly measured coords
            tracked_keypoint.measured_coords = tracked_keypoint.filter.update( updated_coords )
            
# -----------------------------FUNCTION---------------------------------

def filter(keypoint_dict):
    # init wrapper
    wrapper = KalmanWrapper()
    # declare measurement dictionary
    measurement = {}
    for key in keypoint_dict.keys():
        # extract current measured coords for keypoint labeled 'key'
        measure = np.array( [ [keypoint_dict[key][0]] , [keypoint_dict[key][1]]] )
        # print("measured coordinate={} for keypoint={}".format(measure, key))

        # adding measured keypoint coords to measurement dictionary
        measurement.update( { key: measure } )
    # batch update all filter
    wrapper.update(measurement)
    # return (hopefully) more stable keypoint coords
    return wrapper.predict()

def draw_keypoint_on_image(image, keypoint_dict):
        image = np.zeros((192, 192, 3), dtype=np.uint8)
        for key in keypoint_dict.keys():
            centro = (int(keypoint_dict[key][0]), int(keypoint_dict[key][1]))
            image = cv2.circle(image, centro, 1, (255, 0, 0), 2)
        return image
