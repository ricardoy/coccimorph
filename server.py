import uuid
import os
from flask import Flask, request, render_template, redirect, url_for, flash, session, send_from_directory
from werkzeug.utils import secure_filename
from coccimorph.segment import segment as coccimorph_segment
from coccimorph.classifier import predict as coccimorph_predict
import uuid


UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = set(['bmp', 'tif', 'png', 'jpg', 'jpeg', 'gif'])
FOWL = 'FOWL'
RABBIT = 'RABBIT'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = '\x03\xb4TG T5*D\x86\x1c\xd5\xa5V\xbdn\xdd\xa8\xe9\xeff\xad\xb3S'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_full_path():
    return os.path.join(app.config['UPLOAD_FOLDER'], str(session['uid']))


def render_classify_page():
    return render_template('classify.html',
                           filename=session['filename'],
                           uid=session['uid'],
                           threshold=session['threshold'],
                           species=session['species'],
                           scale=session['scale'],
                           suffix=uuid.uuid4().__str__(),
                           classification=session['classification']
                           )


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(get_full_path(), filename)


@app.route('/')
def index():
    if 'uid' not in session:
        session['uid'] = uuid.uuid4()
        app.logger.info('new user: ' + get_full_path())

    if 'filename' not in session:
        session['filename'] = None

    if 'threshold' not in session:
        session['threshold'] = 150

    if 'species' not in session:
        session['species'] = FOWL

    if 'scale' not in session:
        session['scale'] = 11.

    if 'classification' not in session:
        session['classification'] = None

    return render_classify_page()

@app.route('/upload_image', methods=['POST'])
def upload_image():
    app.logger.info('teste')
    if not os.path.exists(get_full_path()):
        try:
            os.makedirs(get_full_path())
        except OSError as e:
            flash(e)
            return redirect(request.url)

    # session['threshold'] = request.form['threshold']
    # session['scale'] = request.form['scale']
    # session['species'] = request.form['species']

    if request.method == 'POST':
        if 'oocyst_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['oocyst_file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(get_full_path(), 'raw_' + filename))
            session['filename'] = filename

            segment(os.path.join(get_full_path(), 'raw_' + filename),
                    session['threshold'],
                    os.path.join(get_full_path(), 'bin_' + filename),
                    os.path.join(get_full_path(), 'seg_' + filename),
                    session['scale'])

    return render_classify_page()


@app.route('/preproc', methods=['POST'])
def preproc():
    session['threshold'] = request.form['threshold']
    session['scale'] = request.form['scale']
    session['species'] = request.form['species']

    # app.logger.debug(request)

    if session['filename'] is not None:
        filename = session['filename']
        segment(os.path.join(get_full_path(), 'raw_' + filename),
                session['threshold'],
                os.path.join(get_full_path(), 'bin_' + filename),
                os.path.join(get_full_path(), 'seg_' + filename),
                session['scale'])
    return render_classify_page()


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    print('oi')
    if session['species'] == FOWL:
        fowl = True
        rabbit = False
    else:
        fowl = False
        rabbit = True

    scale = session['scale']
    if scale is not None and scale != '':
        scale = int(float(scale))
    else:
        scale = None

    classification = coccimorph_predict(
        os.path.join(get_full_path(), 'raw_' + session['filename']),
        int(session['threshold']),
        scale,
        fowl,
        rabbit,
        True
    )
    session['classification'] = classification

    return render_classify_page()


def segment(filename, threshold, binfile, segfile, scale):
    if scale is not None and scale != '':
        scale = int(float(scale))
    else:
        scale = None

    threshold = int(threshold)

    # app.logger.debug('scale', scale)
    # app.logger.debug('threshold', threshold)
    print('scale: ', scale)
    print('threshold: ', threshold)

    coccimorph_segment(filename,
                       threshold,
                       binfile,
                       segfile,
                       scale)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)