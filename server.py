import uuid
import os
import cv2
from flask import Flask, request, render_template, redirect, url_for, flash, session, send_from_directory
from werkzeug.utils import secure_filename
from coccimorph.segment import segment as coccimorph_segment
from coccimorph.classifier import predict as coccimorph_predict
import uuid


UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = set(['bmp', 'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff'])
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


def render_index_page():
    return render_template('index.html')


def render_classify_page():
    return render_template('classify.html',
                           filename=session['filename'],
                           uid=session['uid'],
                           threshold=session['threshold'],
                           species=session['species'],
                           scale=session['scale'],
                           suffix=uuid.uuid4().__str__(),
                           classification=session['classification'],
                           probability=session['probability'],
                           similarity=session['similarity']
                           )


@app.route('/')
def index():
    return render_index_page()


@app.route('/uploads/<filename>')
def send_file(filename):
    '''
    :param filename: the image filename
    :return: the image data to te viewed by the web browser
    '''
    return send_from_directory(get_full_path(), filename)


@app.route('/coccimorph')
def main():
    if 'uid' not in session:
        session['uid'] = uuid.uuid4()
        app.logger.info('new user: ' + get_full_path())

    session['filename'] = None
    session['threshold'] = 150
    session['species'] = FOWL
    session['scale'] = 11.
    session['classification'] = None
    session['similarity'] = None
    session['probability'] = None

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

    if request.method == 'POST':
        if 'oocyst_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['oocyst_file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # print(type(file))
        # print(dir(file))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(get_full_path(), 'raw_' + filename))

            if is_tiff(filename):
                img = cv2.imread(os.path.join(get_full_path(), 'raw_' + filename))
                filename = '%s.bmp' % (filename)
                # print(filename)
                cv2.imwrite(os.path.join(get_full_path(), 'raw_' + filename), img)

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
    if request.form['species'] == FOWL:
        fowl = True
        rabbit = False
        session['species'] = FOWL
    else:
        fowl = False
        rabbit = True
        session['species'] = RABBIT

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
    session['similarity'] = sorted([d for d in classification['similarity'].items()], key=lambda x: x[1], reverse=True)
    session['probability'] = sorted([d for d in classification['probability'].items()], key=lambda x: x[1], reverse=True)

    return render_classify_page()


def segment(filename, threshold, binfile, segfile, scale):
    if scale is not None and scale != '':
        scale = int(float(scale))
    else:
        scale = None

    threshold = int(threshold)

    # print('scale: ', scale)
    # print('threshold: ', threshold)

    coccimorph_segment(filename,
                       threshold,
                       binfile,
                       segfile,
                       scale)


def is_tiff(filename: str):
    f = filename.lower()
    return f.endswith('.tif') or f.endswith('.tiff')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)