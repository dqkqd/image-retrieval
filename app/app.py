import os
import io
import tempfile
import shutil
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, render_template, url_for, request, redirect, Markup
from utils import compactCode, Search
search_model = compactCode(centers_path="../centers/centers.h5py",
                           pq_centers_path="../centers/pq_centers.h5py",
                           codes_path="../centers/codes",
                           codes_name="../centers/codes_name")
model = Search(search_model)

# result
res_img = ['temp/result{:02}.jpg'.format(i) for i in range(10)]
# result combined
com_img = ['temp/combined{:02}.jpg'.format(i) for i in range(10)]

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


MAX_WIDTH = 800
MAX_HEIGHT = 550

def allowed(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def render(idx, name, distance, inlier):
    res = \
        '''
        <div class="row2">
            <div class="block2">
                <h1></h1>
                <span class="text">{}</span>
                <span class="attr">
                    <br>name: {}</br>
                    <br>distance: {:08}</br>
                    <br>inlier: {}</br>
                </span>
                <img class="small" src="{}"/>
                <img class="small" src="{}"/>
                <h1></h1>
            </div>
        </div>
        '''.format(idx+1, name[idx], distance[idx], inlier[idx],
                   url_for('static', filename=res_img[idx%10]),
                   url_for('static', filename=com_img[idx%10]),
                   )

    return res


@app.route("/", methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        filename = request.files['file'].filename
        if filename == "":
            return redirect(request.url)
        else:
            file_image = request.files['file']

            image = Image.open(io.BytesIO(file_image.stream.read()))

            width, height = image.size
            variable = '-'.join([file_image.filename, str(width), str(height)])

            image.save('static/temp/image-search.jpg')

            return redirect(url_for('search', filename=variable))
    else:
        return render_template("home.html")

@app.route("/search/<filename>", methods=['POST', 'GET'])
def search(filename):
    if request.method == 'POST':

        x = request.form.get('x')
        y = request.form.get('y')
        w = request.form.get('w')
        h = request.form.get('h')

        W, H = filename.split('-')[-2:]
        W, H = int(W), int(H)
        SCALE = max(W / MAX_WIDTH, H / MAX_HEIGHT, 1.0)
        if x == "":
            coords = [0, 0, W, H]
        else:
            coords = [int(int(t) * SCALE) for t in [x, y, w, h]]

        model.search('static/temp/image-search.jpg', coords)

        return redirect(url_for("results", pagenum=1))
    else:
        return render_template("search.html", filename='temp/image-search.jpg')


@app.route("/results/<pagenum>", methods=['POST', 'GET'])
def results(pagenum):
    pagenum = int(pagenum)
    model.draw_box(pagenum)

    name = model.rank_list
    distance = model.distance
    inlier = model.inlier

    out = '\n'.join([render(idx, name, distance, inlier) for idx in range((pagenum-1)*10, pagenum*10)])

    return render_template("results.html", out=Markup(out), pagenum=pagenum)


if __name__ == "__main__":
    app.run(debug=True)
