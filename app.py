from flask import Flask, render_template, request
import os
import pandas as pd
from iris_classification import classify_dataset
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # For demo: assume user always wants to predict the last column
        df = pd.read_csv(filepath)
        target_col = df.columns[-1]
        
        try:
            accuracy, confusion_matrix_df, correct, missed = classify_dataset(filepath, target_col)
        except Exception as e:
            return f"Error during classification: {e}"
        
        cm_html = confusion_matrix_df.to_html(classes="table table-bordered", index=True)
        
        return render_template('result.html', accuracy=accuracy, confusion_matrix_html=cm_html, correct=correct, missed=missed)
    
    return "Invalid file format. Please upload a CSV."


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
