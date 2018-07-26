from flask import render_template, request, send_file,session
import os
import app.predict as predict
import app.addingNumber as addingNumber
from flask import flash, redirect, url_for
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.urls import url_parse
from app import app, db
from app.forms import LoginForm
from app.models import User


@app.route('/login', methods=['GET', 'POST'])
def login():
    # if 'username' in session:
    #     return redirect(url_for('index'))
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))

        login_user(user)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')

        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


path = os.getcwd() + '/app/output/'


@app.route("/", methods=["POST", "GET"])
@app.route('/index/<username>')
@login_required
def index():
    for file in os.listdir(path):
        os.remove(path+file)

    args = {"method": "GET"}
    if request.method == "POST":
        file = request.files["file"]
        algorithm = request.form['options']
        if (algorithm == 'predict'):
            if bool(file.filename) and (file.filename.rsplit('.', 1)[1].lower() == 'xlsx'):
                predict.main(file)
                handle = open("app/reason_del.txt", "r", encoding="utf-8")
                data = handle.read()
                args["data"] = data
        elif (algorithm == 'optimizer'):
            addingNumber.write()

        args["method"] = "POST"

    return render_template("index.html", args=args)


@app.route("/downloadExcelFile")
@login_required
def getExcelFile():
    files = os.listdir(path)
    with open(path + files[0]) as fp:
        name = fp.name

    return send_file(name,
                     mimetype='text/xlsx',
                     attachment_filename='output.xlsx',
                     as_attachment=True)

#
# if __name__ == "__main__":
#    app.run(threaded=True)

