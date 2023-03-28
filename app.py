import os
import uuid
import cv2
from flask import Flask, flash, request, redirect, render_template, jsonify
from flask_mysqldb import MySQL
from document_scanner import document_scanner

app = Flask(__name__, static_url_path='/static')

# Configure MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'flask'

# Initialize MySQL
mysql = MySQL(app)

app.secret_key = "secret key"  # for encrypting the session

# It will allow below 16MB contents only, you can change it
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# file Upload
path = os.getcwd() + '/static'
UPLOAD_FOLDER = os.path.join(path, 'uploads')

# Make directory if "uploads" folder not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'tiff'])

# Allow files
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Add this function to save the temporary file
def save_temp_file(file):
    temp_folder = os.path.join(app.config['UPLOAD_FOLDER'], "temp")
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)

    temp_filename = str(uuid.uuid4()) + '.png'
    temp_filepath = os.path.join(temp_folder, temp_filename)
    file.save(temp_filepath)
    return temp_filepath

# Routes
@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist('files[]')

        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        for file in files:
            if file and allowed_file(file.filename):
                name = request.form['inputName']
                # Save the temporary file
                temp_filepath = save_temp_file(file)  
                # Process the uploaded file with the document scanner
                _, _, _, warped, _ = document_scanner(temp_filepath)
                filename = str(uuid.uuid4()) + '.png'
                output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                # Save the processed image
                if warped is not None:
                    cv2.imwrite(output_filepath, warped)
                    # Clean up the temporary file
                    os.remove(temp_filepath)
                    #SQL
                    cur = mysql.connection.cursor()
                    cur.execute(
                        'INSERT INTO flask_test (form_filename, form_name) VALUES (%s,  %s)', (filename, name))
                    mysql.connection.commit()
                    cur.close()
                    flash('Files successfully uploaded.')
                else:
                    flash('Cannot save the image as warped is empty.')

        return redirect('/')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['searchname']
        cur = mysql.connection.cursor()
        cur.execute(
            "SELECT * FROM flask_test WHERE form_name LIKE %s", ('%' + query + '%',))
        data = cur.fetchall()
        return render_template('search.html', data = data)

    return render_template('search.html')

@app.route('/delete', methods=['POST'])
def delete_records():
    record_ids = request.form.getlist('record_ids')

    if record_ids:
        cur = mysql.connection.cursor()

        for record_id in record_ids:
            cur.execute('SELECT form_filename FROM flask_test WHERE form_id = %s', (record_id,))
            result = cur.fetchone()
            if result:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], result[0])
                if os.path.exists(file_path):
                    os.remove(file_path)

            cur.execute('DELETE FROM flask_test WHERE form_id = %s', (record_id,))

        mysql.connection.commit()
        cur.close()

        flash('Selected records and files have been deleted successfully')

    return redirect('/search')

@app.route('/get_admin_password')
def get_admin_password():
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT admin_password FROM admin")
    result = cursor.fetchone()
    if result:
        return jsonify({'password': result[0]})
    else:
        return jsonify({'password': None})

@app.route('/results')
def result():
    cur = mysql.connection.cursor()
    cur.execute('SELECT * FROM flask_test')
    flask_all = cur.fetchall()
    return render_template('result.html', flask_all = flask_all)

@app.route('/edit', methods=['GET', 'POST'])
def edit():
    if request.method == 'POST':
        query = request.form['id']
        cur = mysql.connection.cursor()
        cur.execute(
            "SELECT * FROM flask_test WHERE form_id LIKE %s", ('%' + query + '%',))
        data = cur.fetchall()
        return render_template('edit.html', data = data)

    return render_template('edit.html')

@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def update(id):
    if request.method == 'POST':
        name = request.form['name']
        cur = mysql.connection.cursor()
        cur.execute("UPDATE flask_test SET form_name = %s WHERE form_id = %s", (name, id,))
        mysql.connection.commit()
        cur.close()
        return render_template('edit.html')

if __name__ == "__main__":
    app.run(debug=True)
