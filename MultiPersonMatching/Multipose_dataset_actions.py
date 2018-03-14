from scipy.constants.codata import precision

import multiperson_match
import parse_openpose_json
import numpy as np
import logging
import prepocessing
import glob
import os
import calcAngle
import logging
logger = logging.getLogger("pose_match")

multipose = '/media/jochen/2FCA69D53AB1BFF41/dataset/Multipose/fotos/'
poses = '/media/jochen/2FCA69D53AB1BFF41/dataset/Multipose/poses/'
data = '/media/jochen/2FCA69D53AB1BFF41/dataset/Multipose/json/'
poses = '/media/jochen/2FCA69D53AB1BFF41/dataset/Multipose/poses/'
galabal = '/media/jochen/2FCA69D53AB1BFF41/dataset/galabal2018/json/'
galabalfotos = '/media/jochen/2FCA69D53AB1BFF41/dataset/galabal2018/fotos/'
#pose should look like 00100
def find_matches_with(pose):
    if len(pose) == 5 and pose.isdigit():
        model = data+pose+"_keypoints.json"
        model_features = parse_openpose_json.parse_JSON_multi_person(model)
        count = 0
        os.system("mkdir -p "+poses+pose)
        os.system("mkdir -p "+poses+pose+"/json")
        os.system("mkdir -p "+poses+pose+"/jsonfout")
        os.system("mkdir -p "+poses+pose+"/fotos")
        os.system("mkdir -p "+poses+pose+"/fotosfout")
        for json in glob.iglob(data+"*_keypoints.json"):
            logger.info(json)
            input_features = parse_openpose_json.parse_JSON_multi_person(json)
            (result, error_score, input_transform) = multiperson_match.multi_person_ordered(model_features, input_features, True)
            if result == True:
                place = json.split("_keypoints")[0]
                place = place.split("json/")[1]
                place = place+".json"
                os.system("cp "+json+" "+poses+pose+"/json/"+place)
                foto = json.split("_keypoints")[0];
                foto = foto.replace("json","fotos")
                foto = foto +".jpg"
                os.system("cp "+foto+" "+poses+pose+"/fotos/")
                count = count+1
                logger.info("true")
        print "there are "+str(count)+" matches found"

    else:
        print "find_matches_with has wrong input"



def test_script():
    model = data+"00100_keypoints.json"
    model_features = parse_openpose_json.parse_JSON_multi_person(model)
    input = data+"00100_keypoints.json"

    input_features = parse_openpose_json.parse_JSON_multi_person(input)
    (result, error_score, input_transform) = multiperson_match.multi_person_ordered(model_features, input_features, True)

    print result

def check_matches(pose):
    model = poses+pose+"/json/"+pose+".json"
    model_features = parse_openpose_json.parse_JSON_multi_person(model)
    count =0
    for json in glob.iglob(poses+pose+"/json/*.json"):
        input_features = parse_openpose_json.parse_JSON_multi_person(json)
        (result, error_score, input_transform) = multiperson_match.multi_person_ordered(model_features, input_features, True)
        if result == False:
            count = count +1
            print "error at: "+json
    print str(count)+" foto's werden niet meer herkend"







def rename_files():
    i=0
    for json in glob.iglob(galabal+"*_keypoints.json"):
        i = i+1
        os.system("cp "+json+" "+galabal+str(i)+".json")
        foto = json.split("_keypoints")[0];
        foto = foto.replace("json","fotos")
        foto = foto +".jpg"
        os.system("cp "+foto+" "+galabalfotos+str(i)+".jpg")
