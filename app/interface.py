from flask import render_template, request, send_file,session, send_from_directory
import os
import glob
import app.predict_one_model as predict
<<<<<<< HEAD
import app.optimizer1 as optimizer
=======
>>>>>>> f308dd357f9e616f12cb69b79c15d41170bc56b4
import app.addingNumber as addingNumber
from flask import flash, redirect, url_for
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.urls import url_parse
from app import app, db
from app.forms import LoginForm
from app.models import User


@app.route('/login', methods=['GET', 'POST'])
def login():

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
    args = {"method": "GET"}
    if request.method == "POST":
        file = request.files["file"]
        algorithm = request.form['options']
        if (algorithm == 'predict'):
            if bool(file.filename) and (file.filename.rsplit('.', 1)[1].lower() == 'xlsx'):
                predict.main(file, current_user.username)
        elif (algorithm == 'optimizer'):
            ans, all_str_delete = optimizer.main(file, current_user.username)
            print(ans)
            if ans == 165:
                args["error1"] = 'Таблица пустая! Необходимо заполнить выделенные цветом столбцы!'
            elif ans != [] and all_str_delete:
                args["error2"] = 'Для строк ' + str(ans) + ' из входного файла в исторических данных типоразмер с данной маркой стали и/или группой прочности не найден. Работа оптимизатора не возможна, заполните столбцы "Параметры для оптимизации конкретного режима", либо воспользуйтесть моделью предсказания свойств.'
            elif ans != [] and ~all_str_delete:
                args["error3"] = 'Для строк ' + str(ans) + ' из входного файла в исторических данных типоразмер с данной маркой стали и/или группой прочности не найден. Работа оптимизатора не возможна, заполните столбцы "Параметры для оптимизации конкретного режима", либо воспользуйтесть моделью предсказания свойств.'
        args["method"] = "POST"

    return render_template("index.html", args=args)


@app.route("/downloadExcelFile")
@login_required
def getExcelFile():
    list_of_files = glob.glob(path + '*')
    latest_file = max(list_of_files, key=os.path.getctime)
    with open(latest_file) as fp:
        name = fp.name


    # return redirect(url_for('index'))
    return send_file(name,
                     mimetype='text/xlsx',
                     attachment_filename='output.xlsx',
                     as_attachment=True)

<<<<<<< HEAD
=======

>>>>>>> f308dd357f9e616f12cb69b79c15d41170bc56b4

# @app.route("/redirect")
# @login_required
# def redirect():
#     return redirect(url_for('index'))