import os
import uuid
import tensorflow as tf
import numpy as np
import keras
import random
from PIL import Image
from flask import Flask, flash, request, redirect, render_template, jsonify
from flask_mysqldb import MySQL
from keras.models import load_model
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

tf.keras.models.Model
SEED=0
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
from pythainlp.util import *
#import pytesseract
image_width = 150
image_height = 32

model_path = 'model_adam_21000_300.h5'
weights_path = 'model_weights_adam_21000_300.h5'

model = load_model(model_path)
model.load_weights(weights_path)

def num_char():
    max_len = 35
    characters = ['C', 'ณ', 'ร', 'ก', 'า', 'ฉ', 'ฐ', 'ี', 'ซ', 'ต', 'ญ', 'a', 'ฑ', 'ื', '7', 'v', 'ง', 'เ',
                   '2', 'ฒ', 'อ', 'B', 'ฤ', 'ฎ', 'ท', 'ภ', ' ', '๋', 'ฏ', 'e', 'ถ', ',', 'ฃ', 'ข', '5', 'ธ',
                     'บ', '์', 'ๆ', 'โ', 'ฯ', 'o', 'ห', '1', 'พ', '-', 'ค', '0', 'ะ', '8', 'ษ', '้', 'ป', 'c', 'd',
                       '.', 'ไ', 'ฮ', 'น', '่', 'n', 'จ', 'ฆ', 'แ', 'ิ', '3', 'ม', 'ช', '/', '6', '็', 'ุ', 'ฅ', 'ั', '4',
                         'A', 'ใ', '9', 'ฝ', 'ว', 'r', 'ฟ', 'ส', 'ล', 'ู', 'ฬ', 'ำ', 'ผ', 'ย', 'ด', 'ึ', 'ศ']
    xx = collate(characters)
    AUTOTUNE = tf.data.AUTOTUNE
    #สับคำว่ามีตัวพญัญชนะ
    # Mapping characters to integers.
    char_to_num = tf.keras.layers.StringLookup(vocabulary=xx, mask_token=None)
    # Mapping integers back to original characters.
    num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
    
    return num_to_char,max_len,char_to_num

num_to_char,max_len,char_to_num = num_char()

def numpy_to_tensor(array):
    array = tf.convert_to_tensor(array, dtype=tf.float32)
    array = tf.expand_dims(array,2)
    array = distortion_free_resize(array,img_size = (image_width, image_height))
    array = tf.cast(array, tf.float32) / 255.0

    return array

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_len]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)

    return output_text

def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)
    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]
    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2
    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2
    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)

    return image

def crop_fields(img_output):
    crop_field = []
    crop_field.append(img_output[285:285+70, 900:900+140])  # call time (โทรภายใน)
    crop_field.append(img_output[350:350+70, 750:750+270])  # date 1 (วันที่)
    crop_field.append(img_output[612:612+70, 625:625+210])  # message (บันทึกข้อความที่)
    crop_field.append(img_output[612:612+70, 900:900+300])  # date 2 (ลงวันที่)
    crop_field.append(img_output[674:674+70, 600:600+600])  # for 1 (เพื่อ)
    crop_field.append(img_output[737:737+70, 210:210+250])  # for 2 (เพื่อ-ต่อ)
    crop_field.append(img_output[737:737+70, 590:590+355])  # on (ณ)
    crop_field.append(img_output[737:737+70, 1070:1070+120])  # during the day 1 (ระหว่างวันที่)
    crop_field.append(img_output[799:799+70, 210:210+445])  # during the day 2 (ระหว่างวันที่-ต่อ)
    crop_field.append(img_output[985:985+70, 305:305+200])  # money category (หมวดเงิน)
    crop_field.append(img_output[985:985+70, 580:580+680])  # list (รายการ)
    crop_field.append(img_output[1049:1049+70, 285:285+175])  # money (จำนวน)
    crop_field.append(img_output[1049:1049+70, 525:525+460])  # thai language money (จำนวนภาษาไทย)
    crop_field.append(img_output[1410:1410+70, 875:875+230])  # name 1 (ชื่อบน)
    crop_field.append(img_output[1468:1468+70, 865:865+260])  # position 1 (ตำแหน่งบน)
    crop_field.append(img_output[1670:1670+70, 875:875+230])  # name 2 (ชื่อล่าง)
    crop_field.append(img_output[1728:1728+70, 865:865+260])  # position 2 (ตำแหน่งล่าง)

    return crop_field

# Routes
@app.route('/')
def upload_form():
    return render_template('index.html', processed_image=None)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save the temporary file
            temp_filepath = save_temp_file(file)
            # Process the uploaded file with the document scanner
            warped = document_scanner(temp_filepath)
            crop_field_images = crop_fields(warped)
            imgto_tensor = []
            # Convert to grayscale and array to tensor
            for i in range(len(crop_field_images)):
                crop_field_images[i] = Image.fromarray(crop_field_images[i])
                crop_field_images[i] = crop_field_images[i].convert("L")
                imgto_tensor.append(numpy_to_tensor(crop_field_images[i]))
                # Expand dimensions of the image tensor
                imgto_tensor[i] = tf.expand_dims(imgto_tensor[i], 0)

            data_text = []
            for i in range(len(imgto_tensor)):
                preds = model.predict(imgto_tensor[i])
                pred_texts = decode_batch_predictions(preds)
                if pred_texts == "" or not pred_texts:
                    data_text.append("-")
                else:
                    data_text.append(pred_texts)
            
            for i in range(len(data_text)):
                print(data_text[i], i)

            # Clean up the temporary file
            os.remove(temp_filepath)
            if warped is not None:
                # Generate a unique filename
                filename = str(uuid.uuid4()) + '.png'
                # Set the output filepath
                output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                # Save the processed image
                processed_image = Image.fromarray(warped)
                processed_image.save(output_filepath)
                flash('Document scanned.')
            else:
                flash('Cannot scan the image as warped is empty.')

            return render_template('index.html', processed_image=filename, data_text=data_text)

    return render_template('index.html')

@app.route('/save_data', methods=['POST'])
def save_data():
    if request.method == 'POST':
        # Get the form data
        form_filename = request.form.get('processed_image')
        form_name = request.form.get('inputName')

        f_call = request.form.get('callTime')
        f_date = request.form.get('callDate')
        f_msg = request.form.get('message')
        f_msgdate = request.form.get('dateSign')
        f_for1 = request.form.get('for')
        f_for2 = request.form.get('forContinue')
        f_where = request.form.get('on')
        f_between1 = request.form.get('duringTheDay')
        f_between2 = request.form.get('duringTheDayContinue')
        f_category = request.form.get('moneyCategory')
        f_list = request.form.get('list')
        f_money = request.form.get('money')
        f_moneyth = request.form.get('thaiMoney')
        f_signup = request.form.get('nameUpper')
        f_signupposition = request.form.get('positionUpper')
        f_signdown = request.form.get('nameLower')
        f_signdownposition = request.form.get('positionLower')

        # Insert data into the "flask_test" table
        cur = mysql.connection.cursor()
        insert_query = "INSERT INTO flask_test (form_filename, form_name) VALUES (%s, %s)"
        values = (form_filename, form_name)
        cur.execute(insert_query, values)
        form_id = cur.lastrowid  # Retrieve the generated form_id
        mysql.connection.commit()
        cur.close()

        # Insert data into the "forms" table
        cur = mysql.connection.cursor()
        insert_query = "INSERT INTO forms (f_call, f_date, f_msg, f_msgdate, f_for1, f_for2, f_where, f_between1, f_between2, f_category, f_list, f_money, f_moneyth, f_signup, f_signupposition, f_signdown, f_signdownposition, f_img) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        values = (f_call, f_date, f_msg, f_msgdate, f_for1, f_for2, f_where, f_between1, f_between2, f_category, f_list, f_money, f_moneyth, f_signup, f_signupposition, f_signdown, f_signdownposition, form_id)
        cur.execute(insert_query, values)
        mysql.connection.commit()
        cur.close()

        flash('Data saved successfully.')
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

@app.route('/check_admin_password')
def check_admin_password():
    user = request.args.get('user')
    password = request.args.get('password')

    cur = mysql.connection.cursor()
    cur.execute("SELECT admin_name, admin_password FROM admin WHERE admin_name = %s AND admin_password = %s", (user, password))
    result = cur.fetchone()

    if result:
        return jsonify(valid=True)
    else:
        return jsonify(valid=False)
    
@app.route('/results')
def result():
    cur = mysql.connection.cursor()
    cur.execute('SELECT * FROM flask_test')
    flask_all = cur.fetchall()
    return render_template('result.html', flask_all=flask_all)

@app.route('/edit', methods=['GET', 'POST'])
def edit():
    if request.method == 'POST':
        query = request.form['id']
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT flask_test.*, forms.f_img, forms.f_update, forms.f_call, forms.f_date, forms.f_msg, forms.f_msgdate,
            forms.f_for1, forms.f_for2, forms.f_where, forms.f_between1, forms.f_between2, forms.f_category,
            forms.f_list, forms.f_money, forms.f_moneyth, forms.f_signup, forms.f_signupposition,
            forms.f_signdown, forms.f_signdownposition
            FROM flask_test
            LEFT JOIN forms ON flask_test.form_id = forms.f_img
            WHERE flask_test.form_id LIKE %s
        """, ('%' + query + '%',))
        data = cur.fetchall()
        return render_template('edit.html', data=data)

    return render_template('edit.html')

@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def update(id):
    if request.method == 'POST':
        inputName = request.form['inputName']
        callDate = request.form['callDate']
        dateSign = request.form['dateSign']
        forContinue = request.form['forContinue']
        duringTheDay = request.form['duringTheDay']
        moneyCategory = request.form['moneyCategory']
        money = request.form['money']
        nameUpper = request.form['nameUpper']
        nameLower = request.form['nameLower']
        callTime = request.form['callTime']
        message = request.form['message']
        forField = request.form['for']
        onField = request.form['on']
        duringTheDayContinue = request.form['duringTheDayContinue']
        listField = request.form['list']
        thaiMoney = request.form['thaiMoney']
        positionUpper = request.form['positionUpper']
        positionLower = request.form['positionLower']
        
        cur = mysql.connection.cursor()
        cur.execute("UPDATE flask_test SET form_name = %s WHERE form_id = %s", (inputName, id,))
        cur.execute("UPDATE forms SET f_call = %s, f_date = %s, f_msg = %s, f_msgdate = %s, f_for1 = %s, f_for2 = %s, f_where = %s, f_between1 = %s, f_between2 = %s, f_category = %s, f_list = %s, f_money = %s, f_moneyth = %s, f_signup = %s, f_signupposition = %s, f_signdown = %s, f_signdownposition = %s WHERE f_img = %s", 
                    (callTime, callDate, message, dateSign, forField, forContinue, onField, duringTheDay, duringTheDayContinue, moneyCategory, listField, money, thaiMoney, nameUpper, positionUpper, nameLower, positionLower, id,))
        mysql.connection.commit()
        cur.close()
        return render_template('edit.html')

if __name__ == '__main__':
    app.run(debug=True)
