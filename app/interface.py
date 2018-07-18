from flask import render_template, request, Response, send_file, send_from_directory
import os
import app.predict as predict
import app.addingNumber as addingNumber

from app import app


path = os.getcwd() + '/app/output/'

@app.route("/", methods=["POST", "GET"])
def index():
    for file in os.listdir(path):
        os.remove(path+file)

    template_dir = os.getcwd()
    args = {"method": "GET"}
    if request.method == "POST":
        file = request.files["file"]
        algorithm = request.form['options']
        if (algorithm == 'predict'):
            if bool(file.filename) and (file.filename.rsplit('.', 1)[1].lower() == 'xlsx'):
                predict.main(file.filename)
                handle = open("reason_del.txt", "r", encoding="utf-8")
                data = handle.read()
                args["data"] = data
        elif (algorithm == 'optimizer'):
            addingNumber.write()

        args["method"] = "POST"

    # app = Flask(__name__, template_folder=template_dir)
    return render_template("index.html", args=args)

@app.route("/downloadExcelFile")
def getExcelFile():
    files = os.listdir(path)
    with open(path + files[0]) as fp:
        name = fp.name

    return send_file(name,
                     mimetype='text/xlsx',
                     attachment_filename='output.xlsx',
                     as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)



