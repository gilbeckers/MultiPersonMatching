import Algorithmia
import config
# https://algorithmia.com/developers/clients/python/

project_dir = "data://bilgeckers/multiperson_matching"

# Authenticate with your API key
apiKey = config.algorithmia['apiKey'] # Not included in .git, get your own!
# Create the Algorithmia client object
client = Algorithmia.client(apiKey)
# Instantiate a DataDirectory object, set your data URI and call create
multiperson_matching_directory = client.dir(project_dir)

def upload_pose_img(img_name):
    # Create your data collection if it does not exist
    if multiperson_matching_directory.exists() is False:
        multiperson_matching_directory.create()

    #create a variable that holds the path to the data collection and the img file
    img_file = project_dir + "/" + img_name

    if client.file(img_file).exists() is False:
        # Upload local file
        client.file(img_file).putFile("data/image_data/" + img_name)

def download_pose_img(img_name):
    img_file = project_dir + "/" + img_name
    # Download contents of file as a string
    if client.file(img_file).exists() is True:
        input = client.file(img_file).ur

        #input = client.file(img_file).getFile()
        print(input)

def call_pose_estimation_img_on_server(img_name):
    img_file = project_dir + "/" + img_name
    input = {
        "img_file": img_file
    }
    algo = client.algo('bilgeckers/OpTFpy3_v2/1.0.4')
    result = algo.pipe(input).result
    print(result)
    return result

#upload_pose_img("p2.jpg")
#call_pose_estimation_img_on_server("p2.jpg")

