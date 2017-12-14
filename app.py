from flask import Flask,render_template, request, redirect, url_for,jsonify
from redis import Redis
from werkzeug.utils import secure_filename
import os
import json
import readOpenPoseJson
import pose_matcher
import glob
import base64

UPLOAD_FOLDER = '/home/jochen/fotos/processing'
DOWNLOAD_FOLDER = '/home/jochen/fotos/json/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
redis = Redis(host='127.0.0.1 ', port=6379)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def makeImageInput():
    return render_template('imagePost.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return "no file sended"
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return "no filename"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return processfile(filename)
    return

@app.route('/findmatch', methods=['POST'])
def return_posematch():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return "no file sended"
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return "no filename"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            id = request.form.get('id');
            return findmatch(filename,id)
    return
@app.route('/uploadPose', methods=['POST'])
def add_new_pose():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return "no file sended"
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return "no filename"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            id =0
            for json in glob.iglob("/home/jochen/dataset/poses/*"):
                id = id+1
            filepath ="/home/jochen/dataset/poses/pose"+str(id)
            os.system("mkdir -p " +filepath)
            os.system("mkdir -p " +filepath+"/fotos")
            os.system("mkdir -p " +filepath+"/json")
            os.system("mkdir -p " +filepath+"/Processedfotos")
            filename = filename.rsplit('.')[1]
            filename = "1."+filename
            file.save(os.path.join(filepath+"/fotos", filename))
            os.system("./build/examples/openpose/openpose.bin -write_keypoint_json "+filepath+"/json -image_dir "+filepath+"/fotos -write_images "+filepath+"/Processedfotos -no_display")
            os.system("mv "+filepath+"/json/1_keypoints.json " +filepath+"/json/1.json ")
            return "pose added at id : " +str(id)
    return

@app.route('/getPoses', methods=['GET'])
def return_html_poses():
    return render_template('AllPoses.html')

@app.route('/getAllPoses', methods=['GET'])
def get_all_poses():
    data = []
    count = 0
    for picture in glob.iglob("/home/jochen/dataset/poses/pose*/fotos/1.*"):
        with open(picture, "rb") as image_file:
            dummy ={}
            dummy['naam'] = picture.rsplit('/fotos')[0].rsplit('/poses/')[1]
            encoded_string = base64.b64encode(image_file.read())
            dummy['foto']=encoded_string
            data.append(dummy)
    return json.dumps(data)



def processfile (filename):
    os.system("./build/examples/openpose/openpose.bin -write_keypoint_json /home/jochen/fotos/json -image_dir /home/jochen/fotos/processing -write_images /home/jochen/fotos/calculated_poses/ -no_display")
    os.system("mv -v /home/jochen/fotos/processing/* /home/jochen/fotos/poses")
    filename = filename.rsplit('.')[0]
    filename = filename +'_keypoints.json'
    json_file = open(os.path.join(app.config['DOWNLOAD_FOLDER'], filename), "r")
    json_data = json_file.read()
    return json_data #jsonify(json_data)

def findmatch(filename, id):
    os.system("./build/examples/openpose/openpose.bin -write_keypoint_json /home/jochen/fotos/json -image_dir /home/jochen/fotos/processing -write_images /home/jochen/fotos/calculated_poses/ -no_display")
    os.system("mv -v /home/jochen/fotos/processing/* /home/jochen/fotos/poses")
    filename = filename.rsplit('.')[0]
    inputadr = "/home/jochen/fotos/json/"+filename +'_keypoints.json'
    modeladr = "/home/jochen/dataset/poses/pose"+id+"/json/1.json"
    model = readOpenPoseJson.readJsonFile(modeladr)
    input = readOpenPoseJson.readJsonFile(inputadr)
    #parameters van functie zijn slecht gekozen => secondary = input  &&   primary = model
    result = pose_matcher.is_match(model, input)
    print("Match or not: ", result)
    with open(inputadr) as json_file:
        json_decoded = json.load(json_file)
        json_decoded['match'] = bool(result)
        return jsonify(json_decoded)
    return json_data#jsonify(json_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
