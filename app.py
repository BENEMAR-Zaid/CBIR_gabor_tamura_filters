from flask import Flask, make_response, render_template, request, redirect,jsonify
from lib.index import index_all_gabor, index_one_gabor, index_all_tamura, index_one_tamura
from lib.searcher import Search
from PIL import Image
import numpy as np
import cv2
import os
import shutil
import time
import csv
from paste.translogger import TransLogger
from waitress import serve

app = Flask(__name__)

UPLOAD_FOLDER = 'static/te'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
params = {"theta": 4, "frequency": (0, 1, 0.5, 0.8), "sigma": (1, 3), "n_slice": 2}


#Database Offline Indexing
@app.route('/offlineIndex')
def test():
    index_all_tamura()
    return "Done !!"

#index_all_gabor(params)

@app.route('/')
def cekawal():
	if os.path.exists('static/temp') == True :
		shutil.rmtree('static/temp')
		shutil.rmtree('static/tmp')
		return redirect('/home')
	else :
		return redirect('/home')




@app.route('/home')
def home():


	datasets = os.listdir('static/images')
	if os.path.exists('static/temp') == True :
		image_names = os.listdir('static/temp')
		nearest = sorted(os.listdir('static/temp'))[0]
		target = os.listdir('static/tmp')
		return render_template("index.html", image_names=sorted(image_names),\
		target=(target), page_status=1, count=len(datasets), nearest=(nearest))
	else :
		return render_template("index.html", page_status=2, count=len(datasets))

@app.route('/search', methods=['POST'])
def search():
	file1 = request.files['image']
	file = file1.read()


	npimg = np.frombuffer(file, np.uint8)
	query = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

	
	imgstr = "temp.png"
	cv2.imwrite("static/te/"+ imgstr , query)

	features_gb = index_one_gabor(str(UPLOAD_FOLDER + '/' + imgstr),params)
	features_tm = index_one_tamura(str(UPLOAD_FOLDER + '/' + imgstr))
    #Extracting the feature vetor from the uploaded images and adding this vector to our database
	searcher = Search('conf/index.csv')
	results = searcher.search(features_gb,features_tm)


	
	
	os.makedirs('static/temp')
	os.makedirs('static/tmp')

	i = 1
	for (score, imagePath) in results:
		print(imagePath)
		i += 1
		result = cv2.imread("D:\\Work\\Master\\S3\\Analyse, Mining and Indexing\\Flask\\" + imagePath)
		saveimg = cv2.imwrite("D:\\Work\\Master\\S3\\Analyse, Mining and Indexing\\Flask\\static\\temp\\" + str(score) + str(i) + ".png",result )
		print(result)

	imgstr2 = time.strftime("%Y%m%d-%H%M%S")
	cv2.imwrite("static/tmp/"+ imgstr2 +".jpeg", query)
	return redirect("/home")

@app.route('/<page_name>')
def other_page(page_name):
	return render_template("404.html"), 404

if __name__ == '__main__':
	serve(TransLogger(app, setup_console_handler=False), host="0.0.0.0", port=5000)

