import os
from flask import *  
import numpy as np
import fitz
import re
import pickle
from waitress import serve
import logging
from werkzeug.exceptions import HTTPException

app = Flask(__name__)  
logging.basicConfig(filename='logs.log', level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

@app.route('/check')  
def check():
    """
    Checkpoint
    """  
    return 'We are ready' 

def read_input(input_file):
    """
    Reads input (.pdf) file and returns input_text
    """
    app.logger.info('Reading File')
    with fitz.open(input_file) as doc:
        input_text = ""
        for page in doc:
            input_text += page.get_text()

    # remove extra space
    input_text = re.sub("\s+", " ", input_text)
    input_text = re.sub("/", " ", input_text)
    app.logger.info('returned input text')
    return input_text
 
@app.route('/upload_file', methods = ['POST'])  
def success(): 
    """
    Process the input file and return sentiment
    """ 
    if request.method == 'POST':  
        input_file = request.files['file']
        input_file.save(input_file.filename)
        file_path = os.path.join(os.getcwd(),input_file.filename )
        # read file
        input_text = read_input(file_path) 
        app.logger.info('Loading model to compare the results')
        model = pickle.load(open('sent_model.pkl','rb'))
        candidate_labels = ["positive", "negative", "neutral"]
        result = model(input_text, candidate_labels)
        # remove file
        os.remove(file_path)
        app.logger.info('returned results')
        output = { 
            "doc_sentiment" : result['labels'][0],
            "score" : result['scores'][0]
        }
        return jsonify(output)  
  
if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000, threads=4)