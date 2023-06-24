# -----------------------------IMPORT-----------------------------------
import numpy as np
import cv2

# -----------------------------CLASSES----------------------------------

## Object class for keeping track of detected objects (ID, position, ...relevant info)
class Object(object):

    def __init__(self, detect, ID):
        
        self.prediction = np.asarray(detect)
        self.object_id = ID 
        self.KF = KalmanFilter2D(0.1, 1, 1, 1, 0.1, 0.1)
        self.skip_count = 0 
        self.line = [] 

## Object tracker class that tracks detected object, runs Kalman filter + Khun-Munkres (Hungarian) algorithm
class Tracker(object):

    def __init__(self, min_dist, max_skip, line_length, object_id = 1):
        self.min_dist  = min_dist 
        self.max_skip = max_skip
        self.line_length = line_length
        self.objects = []
        self.object_id = object_id

    def Update(self, detections):
        if self.objects ==[]:
            for i in range(len(detections)):
                self.objects.append( Object(detections[i], self.object_id) )
                self.object_id += 1
                
        N , M = len(self.objects), len(detections)
        cost_matrix = np.zeros(shape=(N, M)) 
        for i in range(N):
            for j in range(M):
                diff = self.objects[i].prediction - detections[j]
                cost_matrix[i][j] = np.sqrt(diff[0][0]*diff[0][0] +diff[1][0]*diff[1][0])
        cost_matrix = (0.5) * cost_matrix 

        assign = []
        for _ in range(N):
            assign.append(-1)
            
        rows, cols = get_minimum_cost_assignment(cost_matrix)
        for i in range(len(rows)):
            assign[rows[i]] = cols[i]

        unassign = []
        for i in range(len(assign)):
            if (assign[i] != -1):
                if (cost_matrix[i][assign[i]] > self.min_dist):
                    assign[i] = -1
                    unassign.append(i)
            else:
                self.objects[i].skip_count += 1

        deleted_objects = []
        for i in range(len(self.objects)):
            if (self.objects[i].skip_count > self.max_skip):
                deleted_objects.append(i)
        if len(deleted_objects) > 0: 
            for id in deleted_objects:
                if id < len(self.objects):
                    del self.objects[id]
                    del assign[id]         

        for i in range(len(detections)):
                if i not in assign:
                    self.objects.append( Object( detections[i], self.object_id )  )
                    self.object_id += 1
                
        for i in range(len(assign)):
            self.objects[i].KF.predict()
            if(assign[i] != -1):
                self.objects[i].skip_count = 0
                self.objects[i].prediction = self.objects[i].KF.update( detections[assign[i]] )
            else:
                self.objects[i].prediction = self.objects[i].KF.update( np.array([[0], [0]]) )
            if(len(self.objects[i].line) > self.line_length):
                for j in range( len(self.objects[i].line) - self.line_length):
                    del self.objects[i].line[j]
            self.objects[i].line.append(self.objects[i].prediction)
            self.objects[i].KF.lastResult = self.objects[i].prediction

## 2-Dimensional Kalman Filter class
class KalmanFilter2D(object):
    
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

## Khun-munkres (Hungarian) algorithm initialization class
class MunkresAlgorithm(object):

    def __init__(self, arr_costs):
        self.X = arr_costs.copy()

        n, m = self.X.shape
        self.u_row = np.ones(n, dtype=bool)
        self.u_column = np.ones(m, dtype=bool)
        self.r_0Z = 0
        self.c_0Z = 0
        self.course = np.zeros((n + m, 2), dtype=int)
        self.check = np.zeros((n, m), dtype=int)

    def clear(self):
        self.u_row[:] = True
        self.u_column[:] = True

# -----------------------------FUNCTION----------------------------------

# Step 1 - Khun-Munkres (Hungarian) algorithm
def row_reduction(assignment):
    assignment.X -= assignment.X.min(axis=1)[:, np.newaxis]
    for i, j in zip(*np.where(assignment.X == 0)):
        if assignment.u_column[j] and assignment.u_row[i]:
            assignment.check[i, j] = 1
            assignment.u_column[j] = False
            assignment.u_row[i] = False

    assignment.clear()
    return cover_columns

# Step 2 - Khun-Munkres (Hungarian) algorithm
def cover_columns(assignment):
    check = (assignment.check == 1)
    assignment.u_column[np.any(check, axis=0)] = False

    if check.sum() < assignment.X.shape[0]:
        return cover_zeros

# Step 3 - Khun-Munkres (Hungarian) algorithm
def cover_zeros(assignment):
    X = (assignment.X == 0).astype(int)
    covered = X * assignment.u_row[:, np.newaxis]
    covered *= np.asarray(assignment.u_column, dtype=int)
    n = assignment.X.shape[0]
    m = assignment.X.shape[1]

    while True:
        row, col = np.unravel_index(np.argmax(covered), (n, m))   
        if covered[row, col] == 0:
            return generate_zeros
        else:
            assignment.check[row, col] = 2
            star_col = np.argmax(assignment.check[row] == 1)
            if assignment.check[row, star_col] != 1:
                assignment.r_0Z = row
                assignment.c_0Z = col
                count = 0
                course = assignment.course
                course[count, 0] = assignment.r_0Z
                course[count, 1] = assignment.c_0Z

                while True:
                    row = np.argmax(assignment.check[:, course[count, 1]] == 1)
                    if assignment.check[row, course[count, 1]] != 1:
                        break
                    else:
                        count += 1
                        course[count, 0] = row
                        course[count, 1] = course[count - 1, 1]

                    col = np.argmax(assignment.check[course[count, 0]] == 2)
                    if assignment.check[row, col] != 2:
                        col = -1
                    count += 1
                    course[count, 0] = course[count - 1, 0]
                    course[count, 1] = col

                for i in range(count + 1):
                    if assignment.check[course[i, 0], course[i, 1]] == 1:
                        assignment.check[course[i, 0], course[i, 1]] = 0
                    else:
                        assignment.check[course[i, 0], course[i, 1]] = 1

                assignment.clear()
                assignment.check[assignment.check == 2] = 0
                return cover_columns
            else:
                col = star_col
                assignment.u_row[row] = False
                assignment.u_column[col] = True
                covered[:, col] = X[:, col] * (
                    np.asarray(assignment.u_row, dtype=int))
                covered[row] = 0

# Step 4 - Khun-Munkres (Hungarian) algorithm
def generate_zeros(assignment):
    if np.any(assignment.u_row) and np.any(assignment.u_column):
        minimum_value = np.min(assignment.X[assignment.u_row], axis=0)
        minimum_value = np.min(minimum_value[assignment.u_column])
        assignment.X[~assignment.u_row] += minimum_value
        assignment.X[:, assignment.u_column] -= minimum_value
    return cover_zeros

# Assignment function
def get_minimum_cost_assignment(arr_costs):
    arr_costs = np.asarray(arr_costs)
        
    if arr_costs.shape[1] < arr_costs.shape[0]:
        arr_costs = arr_costs.T
        is_T = True
    else:
        is_T = False

    assignment = MunkresAlgorithm(arr_costs)
    
    run = None if 0 in arr_costs.shape else row_reduction

    while run is not None:
        run = run(assignment)

    if is_T:
        check = assignment.check.T
    else:
        check = assignment.check
    return np.where(check == 1)

# Visual tracking feedback
def draw_line(tracker_object, image):
    # print('before outer for')
    for i in range(len(tracker_object.objects)):
        # print('before if')
        if (len(tracker_object.objects[i].line) > 1):
            print('before inner for')
            for j in range( len(tracker_object.objects[i].line) - 1):
                x1 = tracker_object.objects[i].line[j][0][0]
                y1 = tracker_object.objects[i].line[j][1][0]
                x2 = tracker_object.objects[i].line[j+1][0][0]
                y2 = tracker_object.objects[i].line[j+1][1][0]
                id = tracker_object.objects[i].object_id
                # print("x1={} y1={} x2={} y2={} id={}", x1, y1, x2, y2, id)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            cv2.putText(image, str(id), (int(x1)-10, int(y1)-20), 0, 0.5, (0,0,255), 2)

# Extract wrists keypoints coordinates 
def get_wrist_keypoints(keypoint_dict):
    keypoint_coordinates = []
    for key in keypoint_dict.keys():
        if key == 9 or key == 10:
            coordinate = np.array( [ [keypoint_dict[key][0]] , [keypoint_dict[key][1]]] )
            keypoint_coordinates.append(np.round(coordinate))
    return keypoint_coordinates

# Visual tracking feedback (circle)
def draw_circle(keypoints_coordinate, image):
    for coordinate in keypoints_coordinate:
        (x, y) = coordinate
        centeroid = (int(x), int(y))
        cv2.circle(image, centeroid, 5, (0, 0, 255), 2)

# Kalman filter for more stable keypoint coords
def kalman_filter(keypoint_dict):
    # step, acceleration_x (m/s^2), acceleration_y (m/s^2), std_dev_acceleration (m/s^2), x_std_dev_measurement (m), y_std_dev_measurement (m)
    filter = KalmanFilter2D(1, 0.1, 0.1, 0.1, 0.05, 0.05)
    for key in keypoint_dict.keys():
        measured = np.array( [ [keypoint_dict[key][0]] , [keypoint_dict[key][1]]] )
        # print("measured coordinate=", measured)
        filtered = filter.update(measured)
        # print("filtered coordinate=", filtered)
        predicted = filter.predict()
        # print("predicted coordinate=", predicted)
        keypoint_dict[key][0], keypoint_dict[key][1] = predicted[0], predicted[1] 
    return keypoint_dict




    


# -----------------------------------------------------------------------