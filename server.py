import uuid
import os
from flask import Flask, request, render_template, redirect, url_for, flash, session, send_from_directory
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = set(['bmp', 'tif', 'png', 'jpg', 'jpeg', 'gif'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = '\x03\xb4TG T5*D\x86\x1c\xd5\xa5V\xbdn\xdd\xa8\xe9\xeff\xad\xb3S'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_full_path():
    return os.path.join(app.config['UPLOAD_FOLDER'], str(session['uid']))


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(get_full_path(), filename)


@app.route('/', methods=['GET', 'POST'])
def classify():
    if 'uid' not in session:
        session['uid'] = uuid.uuid4()
        app.logger.info(get_full_path())

    if not os.path.exists(get_full_path()):
        try:
            os.makedirs(get_full_path())
        except OSError as e:
            flash(e)
            return redirect(request.url)

    filename = None
    if request.method == 'POST':
        if 'oocyst_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['oocyst_file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = 'processed_' + secure_filename(file.filename)
            file.save(os.path.join(get_full_path(), filename))

    return render_template('classify.html', filename=filename, uid=session['uid'])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)