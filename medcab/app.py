from os import getenv
from flask import Flask, request, render_template, redirect, jsonify
from flask_login import login_required, current_user, login_user, logout_user
from .loginform import UserModel, db, login

from flask_bootstrap import Bootstrap
from .predict import pred_function

def create_app():
     # constructs core flask app, 
    app = Flask(__name__, template_folder='templates', static_folder='static')
    Bootstrap(app)
    
    app.config["SECRET_KEY"] = getenv('SECRET_KEY')
    app.config['SQLALCHEMY_DATABASE_URI'] = getenv('SQLITE_DATABASE')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    

    from .loginform import db, login
    db.init_app(app)
    login.init_app(app)
    login.login_view = 'login' 
    

    @app.before_first_request
    def create_all():
        db.create_all()
    
    
    @app.route('/', methods=['POST', "GET"])
    @login_required
    def home():
        return render_template('home.html')

    @app.route('/login', methods = ['POST', 'GET'])
    def login():
        if current_user.is_authenticated:
            return redirect('/')
    
        if request.method == 'POST':
            username = request.form['username']
            user = UserModel.query.filter_by(username = username).first()
            if user is not None and user.check_password(request.form['password']):
                login_user(user)
                return redirect('/recomendations')

        return render_template('login.html')
    
    @app.route('/register', methods = ['POST', "GET"])
    def register():
        if current_user.is_authenticated:
            return redirect('/')
    
        if request.method == 'POST':
            email = request.form['email']
            username = request.form['username']
            password = request.form['password']
        
            if UserModel.query.filter_by(email = email).first():
                return('Email Already In Use')
        
            user = UserModel(email=email, username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            return redirect('/login')
        return render_template('register.html')

    @app.route('/logout')
    def logout():
        logout_user()
        return redirect('/')

    @app.route('/recomendations', methods=['POST',"GET"])
    # def recomendations():
    #     return 'todo'
    def recomendations():
        if request.method == 'POST':
            
    
            preds = pred_function(request.form['input'])
            return jsonify(preds)
        return render_template('recomendations.html')
    

    # @app.route('/results')
    # def search_results(search):
    #     results = []
    #     search_string = search.data['search']

    #     if search.data['search'] == '':
    #         qry = db_session.query(marijuana)
    #         results = qry.all()

    #     if not results:
    #         flash('no results found!')
    #         return redirect('/')
    #     else:
    #         return render_template('results.html', results=results)

        
    return app