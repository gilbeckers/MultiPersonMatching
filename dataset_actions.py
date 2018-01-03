from scipy.constants.codata import precision

import pose_match
import parse_openpose_json
import numpy as np
import logging
import prepocessing
import glob
import os

path = '/media/jochen/2FCA69D53AB1BFF49/dataset/poses/'
data = '/media/jochen/2FCA69D53AB1BFF49/dataset/data/'
matched = '/media/jochen/2FCA69D53AB1BFF49/dataset/matched/'

def replace_json_files(pose):
    for foto in glob.iglob(path+pose+"/fotosfout/*"):
        foto = foto.split(".")[0];
        foto = foto.replace("fotosfout","json")
        foto = foto +".json"
        string ="mv "+foto+" "+path+pose+"/jsonfout/"
        os.system("mv "+foto+" "+path+pose+"/jsonfout/")

def find_treshholds(pose):
    #THRESHOLDS
    eucl_dis_tresh_torso = 0.076 #0.065  of 0.11 ??
    rotation_tresh_torso = 28.667
    eucl_dis_tresh_legs = 0.062
    rotation_tresh_legs = 18.678
    eucld_dis_shoulders_tresh = 0.063
    model = path+"pose1/json/1.json"
    model_features = parse_openpose_json.parse_JSON_single_person(model)
    model_features = prepocessing.unpad(model_features)
    for json in glob.iglob(matched+"json/*"):

        input_features = parse_openpose_json.parse_JSON_single_person(json)

        input_features = prepocessing.unpad(input_features)


        (result, error_score, input_transform) = pose_match.single_person_treshholds(model_features, input_features,eucl_dis_tresh_torso,rotation_tresh_torso,eucl_dis_tresh_legs,rotation_tresh_legs,eucld_dis_shoulders_tresh,True)
        while(result==True):
            eucl_dis_tresh_legs = eucl_dis_tresh_legs - 0.001
            print json
            (result, error_score, input_transform) = pose_match.single_person_treshholds(model_features, input_features,eucl_dis_tresh_torso,rotation_tresh_torso,eucl_dis_tresh_legs,rotation_tresh_legs,eucld_dis_shoulders_tresh,True)
    print "best eucl_dis_tresh_torso: " +str(eucl_dis_tresh_legs)

def check_pose_data(pose):
    #THRESHOLDS
    eucl_dis_tresh_torso = 0.076 #0.065  of 0.11 ??
    rotation_tresh_torso = 28.667
    eucl_dis_tresh_legs = 0.062
    rotation_tresh_legs = 18.678
    eucld_dis_shoulders_tresh = 0.063
    model = path+pose+"/json/1.json"
    model_features = parse_openpose_json.parse_JSON_single_person(model)
    model_features = prepocessing.unpad(model_features)
    for json in glob.iglob(path+pose+"/json/*"):

        input_features = parse_openpose_json.parse_JSON_single_person(json)
        input_features = prepocessing.unpad(input_features)

        (result, error_score, input_transform) = pose_match.single_person_treshholds(model_features, input_features,eucl_dis_tresh_torso,rotation_tresh_torso,eucl_dis_tresh_legs,rotation_tresh_legs,eucld_dis_shoulders_tresh,True)
        if result == False:
            print "error at" +json

def sort_pose_from_data(model):
    model_features = parse_openpose_json.parse_JSON_single_person(model)
    model_features = prepocessing.unpad(model_features)
    number =0
    for json in glob.iglob(data+"set*/a/json/*"):
        input_features = parse_openpose_json.parse_JSON_single_person(json)
        input_features = prepocessing.unpad(input_features)
        (result, error_score, input_transform) = pose_match.single_person(model_features, input_features, True)
        if result ==True:
            number = number +1

            os.system("cp "+json+" "+matched+"/json")
            foto = json.split("_")[0]
            foto = foto.replace("json","image")
            foto = foto +".png"
            os.system("cp "+foto+" "+matched+"/image")

    print "number is "+str(number)

def check_pose_score(pose):
    check_pose_data(pose)
    model = path+pose+"/json/1.json"
    model_features = parse_openpose_json.parse_JSON_single_person(model)
    model_features = prepocessing.unpad(model_features)
    number = 0
    for json in glob.iglob(path+"*/json/*"):
        if json.split("/json/")[0].split("/poses/")[1] != pose:
            input_features = parse_openpose_json.parse_JSON_single_person(json)
            input_features = prepocessing.unpad(input_features)
            (result, error_score, input_transform) = pose_match.single_person(model_features, input_features, True)
            if result ==True:
                number = number +1

                os.system("cp "+json+" "+matched+"/json")
                foto = json.split(".json")[0];
                foto = foto.replace("json","fotos")
                foto = foto +".png"
                os.system("cp "+foto+" "+matched+"/image")
    print "number of falses: " + str(number)
