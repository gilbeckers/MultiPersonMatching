import numpy as np

def calculateAngleOf3Points(point1 , point2, point3):
    ba = (point1[0][0] -point2[0][0],point1[0][1] -point2[0][1],)
    bc = (point3[0][0] -point2[0][0],point3[0][1] -point2[0][1],)
    cosine_angle = 0
    if  np.linalg.norm(ba) * np.linalg.norm(bc) != 0:
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    #print np.degrees(angle)
    return np.degrees(angle)

def prepareangles(points):
    result = np.array([])
    #angle of right shoulder
    right_shoulder = calculateAngleOf3Points(points[1:2],points[2:3],points[3:4])
    result = np.append(result,right_shoulder)
    #print("right shoulder: ", right_shoulder)
    #angle of left shoulder
    left_shoulder =calculateAngleOf3Points(points[1:2],points[5:6],points[6:7])
    result = np.append(result,left_shoulder)
    #print("left shoulder: ", left_shoulder)
    #angle of right elbow
    right_elbow =calculateAngleOf3Points(points[2:3],points[3:4],points[4:5])
    result = np.append(result,right_elbow)
    #print("right elbow: ", right_elbow)
    #angle of left elbow
    left_elbow =calculateAngleOf3Points(points[5:6],points[6:7],points[7:8])
    result = np.append(result,left_elbow)
    #print("left elbow: ", left_elbow)
    #angle of right hip
    right_hip =calculateAngleOf3Points(points[2:3],points[8:9],points[9:10])
    result = np.append(result,right_hip)
    #print("right hip: ", right_hip)
    #angle of left hip
    left_hip =calculateAngleOf3Points(points[5:6],points[11:12],points[12:13])
    result = np.append(result,left_hip)
    #print("left hip: ", left_hip)
    #angle of right knee
    right_knee =calculateAngleOf3Points(points[8:9],points[9:10],points[10:11])
    result = np.append(result,right_knee)
    #print("right knee: ", right_knee)
    #angle of left knee
    left_knee =calculateAngleOf3Points(points[11:12],points[12:13],points[13:14])
    result = np.append(result,left_knee)
    #print("left knee: ", left_knee)
    #print("\n")
    return result

def compare(prim,second):
    max_error = 0
    for i in range(0, prim.size):
        if abs(prim[i:i+1]-second[i:i+1])>max_error:
            max_error = abs(prim[i]-second[i])
    print("max error is : ", max_error)
    return max_error

def succes(prim,second):
    angles = 0
    for i in range(0, prim.size):
            angles = np.append(angles,abs(prim[i]-second[i]))

    succes = True
    for angle in angles:
        if angle > 20:
            succes = False
    return succes

def find_treshhold(prim,second,tresh):
        angles = 0
        for i in range(0, prim.size):
                angles = np.append(angles,abs(prim[i]-second[i]))

        succes = True
        for angle in angles:
            if angle > tresh:
                succes = False
        return succes
