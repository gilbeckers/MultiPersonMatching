from scipy.constants.codata import precision

import pose_match
import parse_openpose_json
import numpy as np
import logging
import prepocessing
import glob
import os
import calcAngle

path = '/media/jochen/2FCA69D53AB1BFF49/dataset/poses/'
data = '/media/jochen/2FCA69D53AB1BFF49/dataset/data/'
matched = '/media/jochen/2FCA69D53AB1BFF49/dataset/matched/'

def make_poses_left():
    count = 0
    for json in glob.iglob(data+"set*/*/json/*"):
        os.system("cp "+json+" "+path+"pose0left/json/"+str(count)+".json")
        foto = json.split("_")[0]
        foto = foto.replace("json","image")
        foto = foto +".png"
        os.system("cp "+foto+" "+path+"pose0left/fotos/"+str(count)+".png")
        count = count+1

def make_poses_from_poses_left(pose):
    eucl_dis_tresh_torso = 0.075 #0.065  of 0.11 ??
    rotation_tresh_torso = 16.801
    eucl_dis_tresh_legs = 0.046
    rotation_tresh_legs = 14.588
    eucld_dis_shoulders_tresh = 0.068


    os.system("mkdir -p "+path+pose)
    os.system("mkdir -p "+path+pose+"/json")
    os.system("mkdir -p "+path+pose+"/jsonfout")
    os.system("mkdir -p "+path+pose+"/fotos")
    os.system("mkdir -p "+path+pose+"/fotosfout")
    model = path+pose+"/json/0.json"
    model_features = parse_openpose_json.parse_JSON_single_person(model)
    count =0
    for json in glob.iglob(path+"pose0left/json/*"):
        input_features = parse_openpose_json.parse_JSON_single_person(json)
        (result, error_score, input_transform) = pose_match.single_person_treshholds(model_features, input_features,eucl_dis_tresh_torso,rotation_tresh_torso,eucl_dis_tresh_legs,rotation_tresh_legs,eucld_dis_shoulders_tresh,True)
        if result == True:
            os.system("mv "+json+" "+path+pose+"/json/")
            foto = json.split(".")[0];
            foto = foto.replace("json","fotos")
            foto = foto +".png"
            os.system("mv "+foto+" "+path+pose+"/fotos/")
            count = count+1
    print "there are "+str(count)+" matches found"

def statistics():

    for i in range(1,7):
        affine_right =0
        affine_rightmissed =0
        affine_wrong =0
        angle_right =0
        angle_rightmissed =0
        angle_wrong =0
        affinev2_right =0
        affinev2_rightmissed =0
        affinev2_wrong =0
        total_jsonfout = 0
        pose = "pose"+str(i)
        model = path+pose+"/json/0.json"
        model_features = parse_openpose_json.parse_JSON_single_person(model)
        for json in glob.iglob(path+pose+"/json/*"):
            input_features = parse_openpose_json.parse_JSON_single_person(json)
            (result, error_score, input_transform) = pose_match.single_person(model_features, input_features, True)
            if result == True:
                affine_right +=1
            else:
                affine_rightmissed +=1

            (result, error_score, input_transform) = pose_match.single_person_v2(model_features, input_features, True)
            if result == True:
                affinev2_right +=1
            else:
                affinev2_rightmissed +=1

            primary_angles = calcAngle.prepareangles(model_features)
            secondary_angles = calcAngle.prepareangles(input_features)
            result = calcAngle.succes(primary_angles,secondary_angles)
            if result == True:
                angle_right +=1
            else:
                angle_rightmissed +=1

        for json in glob.iglob(path+pose+"/jsonfout/*"):
            input_features = parse_openpose_json.parse_JSON_single_person(json)
            (result, error_score, input_transform) = pose_match.single_person(model_features, input_features, True)
            if result ==True:
                affine_wrong +=1

            (result, error_score, input_transform) = pose_match.single_person_v2(model_features, input_features, True)
            if result ==True:
                affinev2_wrong +=1

            primary_angles = calcAngle.prepareangles(model_features)
            secondary_angles = calcAngle.prepareangles(input_features)
            result = calcAngle.succes(primary_angles,secondary_angles)
            if result ==True:
                angle_wrong +=1

            total_jsonfout += 1

        print "******************************************************************************"
        print "for "+pose+":"
        print "affine_right = "+str(affine_right)
        print "affine_rightmissed = "+str(affine_rightmissed)
        print "affine_wrong = "+str(affine_wrong)

        print ""
        print "affinev2_right = "+str(affinev2_right)
        print "affinev2_rightmissed = "+str(affinev2_rightmissed)
        print "affinev2_wrong = "+str(affinev2_wrong)

        print ""
        print "angle_right = "+str(angle_right)
        print "angle_rightmissed = "+str(angle_rightmissed)
        print "angle_wrong = "+str(angle_wrong)

        print ""
        print "total_jsonfout = "+str(total_jsonfout)

def clean_dataset():
    #THRESHOLDS

    for i in range(1,9):
        pose = "pose"+str(i)
        os.system("mkdir -p "+path+pose+"/fotosfout")
        os.system("mkdir -p "+path+pose+"/jsonfout")
        model = path+pose+"/json/0.json"
        model_features = parse_openpose_json.parse_JSON_single_person(model)
        count =0
        for json in glob.iglob(path+pose+"/json/*"):
            count =0
            input_features = parse_openpose_json.parse_JSON_single_person(json)
            (result, error_score, input_transform) = pose_match.single_person(model_features, input_features, True)
            if result == False:
                os.system("mv "+json+" "+path+pose+"/jsonfout/")
                foto = json.split(".")[0];
                foto = foto.replace("json","fotos")
                foto = foto +".png"
                os.system("mv "+foto+" "+path+pose+"/fotosfout/")
                count = count+1
        print "moved "+str(count)+" at "+pose

def reorder_json_fout():
    moved = 0
    for i in range(1,58):
        pose = "pose"+str(i)
        for json in glob.iglob(path+pose+"/jsonfout/*"):
            input_features = parse_openpose_json.parse_JSON_single_person(json)
            for x in range(1,58):
                pose2 = "pose"+str(x)
                model = path+pose2+"/json/1.json"
                model_features = parse_openpose_json.parse_JSON_single_person(model)
                (result, error_score, input_transform) = pose_match.single_person(model_features, input_features, True)
                if result == True:
                    os.system("mv "+json+" "+path+pose+"/json/")
                    foto = json.split(".")[0];
                    foto = foto.replace("json","fotos")
                    foto = foto +".png"
                    os.system("mv "+foto+" "+path+pose+"/fotos/")
                    moved = moved +1
    print "finished and moved : "+str(moved)


def replace_json_files(pose):
    for foto in glob.iglob(path+pose+"/fotosfout/*"):
        foto = foto.split(".")[0];
        foto = foto.replace("fotosfout","json")
        foto = foto +".json"
        string ="mv "+foto+" "+path+pose+"/jsonfout/"
        os.system("mv "+foto+" "+path+pose+"/jsonfout/")

def replace_json_files_all():
    for i in range(1,8):
        pose = "pose"+str(i)
        for foto in glob.iglob(path+pose+"/fotosfout/*"):
            foto = foto.split(".")[0];
            foto = foto.replace("fotosfout","json")
            foto = foto +".json"
            string ="mv "+foto+" "+path+pose+"/jsonfout/"
            os.system("mv "+foto+" "+path+pose+"/jsonfout/")

def find_treshholds(pose):
    #THRESHOLDS
    eucl_dis_tresh_torso = 0.65 #0.065  of 0.11 ??
    rotation_tresh_torso = 1000
    eucl_dis_tresh_legs = 1000
    rotation_tresh_legs = 400
    eucld_dis_shoulders_tresh = 100
    angle =0
    model = path+pose+"/json/0.json"
    model_features = parse_openpose_json.parse_JSON_single_person(model)
    replace_json_files(pose)
    print "angle"
    for json in glob.iglob(path+pose+"/json/*"):
        input_features = parse_openpose_json.parse_JSON_single_person(json)
        primary_angles = calcAngle.prepareangles(model_features)
        secondary_angles = calcAngle.prepareangles(input_features)
        result = calcAngle.find_treshhold(primary_angles,secondary_angles,angle)
        while(result==False):
            angle = angle + 0.001
            result = calcAngle.find_treshhold(primary_angles,secondary_angles,angle)
            if result == True:
                print json
    '''
    print 'eucl_dis_tresh_torso'
    eucl_dis_tresh_torso = 0
    for json in glob.iglob(path+pose+"/json/*"):
        input_features = parse_openpose_json.parse_JSON_single_person(json)
        (result, error_score, input_transform) = pose_match.single_person_treshholds(model_features, input_features,eucl_dis_tresh_torso,rotation_tresh_torso,eucl_dis_tresh_legs,rotation_tresh_legs,eucld_dis_shoulders_tresh,True)
        while(result==False):
            eucl_dis_tresh_torso = eucl_dis_tresh_torso + 0.001
            (result, error_score, input_transform) = pose_match.single_person_treshholds(model_features, input_features,eucl_dis_tresh_torso,rotation_tresh_torso,eucl_dis_tresh_legs,rotation_tresh_legs,eucld_dis_shoulders_tresh,True)
            if result == True:
                print json

    print 'rotation_tresh_torso'
    rotation_tresh_torso = 0
    for json in glob.iglob(path+pose+"/json/*"):
        input_features = parse_openpose_json.parse_JSON_single_person(json)
        (result, error_score, input_transform) = pose_match.single_person_treshholds(model_features, input_features,eucl_dis_tresh_torso,rotation_tresh_torso,eucl_dis_tresh_legs,rotation_tresh_legs,eucld_dis_shoulders_tresh,True)
        while(result==False):
            rotation_tresh_torso = rotation_tresh_torso + 0.001
            (result, error_score, input_transform) = pose_match.single_person_treshholds(model_features, input_features,eucl_dis_tresh_torso,rotation_tresh_torso,eucl_dis_tresh_legs,rotation_tresh_legs,eucld_dis_shoulders_tresh,True)
            if result == True:
                print json

    print 'eucl_dis_tresh_legs'
    eucl_dis_tresh_legs = 0
    for json in glob.iglob(path+pose+"/json/*"):
        input_features = parse_openpose_json.parse_JSON_single_person(json)
        (result, error_score, input_transform) = pose_match.single_person_treshholds(model_features, input_features,eucl_dis_tresh_torso,rotation_tresh_torso,eucl_dis_tresh_legs,rotation_tresh_legs,eucld_dis_shoulders_tresh,True)
        while(result==False):
            eucl_dis_tresh_legs = eucl_dis_tresh_legs + 0.001
            (result, error_score, input_transform) = pose_match.single_person_treshholds(model_features, input_features,eucl_dis_tresh_torso,rotation_tresh_torso,eucl_dis_tresh_legs,rotation_tresh_legs,eucld_dis_shoulders_tresh,True)
            if result == True:
                print json

    print 'rotation_tresh_legs'
    rotation_tresh_legs = 0
    for json in glob.iglob(path+pose+"/json/*"):
        input_features = parse_openpose_json.parse_JSON_single_person(json)
        (result, error_score, input_transform) = pose_match.single_person_treshholds(model_features, input_features,eucl_dis_tresh_torso,rotation_tresh_torso,eucl_dis_tresh_legs,rotation_tresh_legs,eucld_dis_shoulders_tresh,True)
        while(result==False):
            rotation_tresh_legs = rotation_tresh_legs + 0.001
            (result, error_score, input_transform) = pose_match.single_person_treshholds(model_features, input_features,eucl_dis_tresh_torso,rotation_tresh_torso,eucl_dis_tresh_legs,rotation_tresh_legs,eucld_dis_shoulders_tresh,True)
            if result == True:
                print json

    print 'eucld_dis_shoulders_tresh'
    eucld_dis_shoulders_tresh = 0
    for json in glob.iglob(path+pose+"/json/*"):
        input_features = parse_openpose_json.parse_JSON_single_person(json)
        (result, error_score, input_transform) = pose_match.single_person_treshholds(model_features, input_features,eucl_dis_tresh_torso,rotation_tresh_torso,eucl_dis_tresh_legs,rotation_tresh_legs,eucld_dis_shoulders_tresh,True)
        while(result==False):
            eucld_dis_shoulders_tresh = eucld_dis_shoulders_tresh + 0.001
            (result, error_score, input_transform) = pose_match.single_person_treshholds(model_features, input_features,eucl_dis_tresh_torso,rotation_tresh_torso,eucl_dis_tresh_legs,rotation_tresh_legs,eucld_dis_shoulders_tresh,True)
            if result == True:
                print json

    print "eucl_dis_tresh_torso= " +str(eucl_dis_tresh_torso )
    print "rotation_tresh_torso= " +str(rotation_tresh_torso)
    print "eucl_dis_tresh_legs= " +str(eucl_dis_tresh_legs)
    print "rotation_tresh_legs= " +str(rotation_tresh_legs)
    print "eucld_dis_shoulders_tresh= " +str(eucld_dis_shoulders_tresh)
    '''
    print "angle = " +str(angle)
def check_pose_data(pose):
    model = path+pose+"/json/1.json"
    model_features = parse_openpose_json.parse_JSON_single_person(model)
    count =0
    for json in glob.iglob(path+pose+"/json/*"):
        input_features = parse_openpose_json.parse_JSON_single_person(json)
        (result, error_score, input_transform) = pose_match.single_person(model_features, input_features, True)
        if result == False:
            count = count+1
    return count

def sort_pose_from_data(model):
    model_features = parse_openpose_json.parse_JSON_single_person(model)
    number =0
    for json in glob.iglob(data+"set*/a/json/*"):
        input_features = parse_openpose_json.parse_JSON_single_person(json)
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
    count = check_pose_data(pose)
    model = path+pose+"/json/1.json"
    model_features = parse_openpose_json.parse_JSON_single_person(model)
    number = 0
    for json in glob.iglob(path+"*/json/*"):
        if json.split("/json/")[0].split("/poses/")[1] != pose:
            input_features = parse_openpose_json.parse_JSON_single_person(json)
            (result, error_score, input_transform) = pose_match.single_person(model_features, input_features, True)
            if result ==True:
                number = number +1

                os.system("cp "+json+" "+matched+"/json")
                foto = json.split(".json")[0];
                foto = foto.replace("json","fotos")
                foto = foto +".png"
                os.system("cp "+foto+" "+matched+"/image")
    print "number of falses: " + str(number)
    print "number of right not detected: " + str(count)
