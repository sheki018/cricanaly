from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for, Markup
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import xgboost
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/fitness", methods=["GET", "POST"])
def fitness():
    return render_template("fitness-home.html")

@app.route("/change", methods=["GET", "POST"])
def change():
    if(request.form['radio']== 'batsmen'):
        return render_template("fitness-batsmen.html")
    elif(request.form['radio']=='bowler'):
        return render_template("fitness-bowler.html")

@app.route("/predict-batsmen", methods=["GET", "POST"])
def predict():

    file = "batsmen_fitness.xlsx"
    df = pd.read_excel(file)

    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(X[:, 1:])
    X[:, 1:] = imputer.transform(X[:, 1:])

    columnTransformer = ColumnTransformer(
        [('encoder', OneHotEncoder(), [0])],     remainder='passthrough')

    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    knn = KNeighborsClassifier()
    xgb = xgboost.XGBClassifier()
    clf = [('dtc', dtc), ('rfc', rfc), ('knn', knn),
           ('xgb', xgb)]  # list of (str, estimator)

    lr = LogisticRegression()
    stack_model = StackingClassifier(estimators=clf, final_estimator=lr)
    pipe = make_pipeline(columnTransformer, stack_model)

    pipe.fit(X,Y)    

    player_names = [[x for x in request.form.values()]]

    my_prediction = pipe.predict(player_names)

    if my_prediction > 0:
        return render_template('fitness-batsmen.html', pred="You live to play another day ,you are fit to play!!!")
    else:
        return render_template('fitness-batsmen.html', pred="Do not go gently into the night, not fit to play!!!")

@app.route("/predict-bowler", methods=["GET", "POST"])
def predictBowler():

    file = "bowlers_fitness.xlsx"
    df = pd.read_excel(file)

    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(X[:, 1:])
    X[:, 1:] = imputer.transform(X[:, 1:])

    columnTransformer = ColumnTransformer(
        [('encoder', OneHotEncoder(), [0])],     remainder='passthrough')

    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    knn = KNeighborsClassifier()
    xgb = xgboost.XGBClassifier()
    clf = [('dtc', dtc), ('rfc', rfc), ('knn', knn),
           ('xgb', xgb)]  # list of (str, estimator)

    lr = LogisticRegression()
    stack_model = StackingClassifier(estimators=clf, final_estimator=lr)
    pipe = make_pipeline(columnTransformer, stack_model)
    pipe.fit(X,Y)    

    player_names = [[x for x in request.form.values()]]

    my_prediction = pipe.predict(player_names)

    if my_prediction > 0:
        return render_template('fitness-bowler.html', pred="You live to fight another day ,you are fit to play!!!")
    else:
        return render_template('fitness-bowler.html', pred="Do not go gently into the night, not fit to play!!!")


@app.route("/mental-health", methods=["GET", "POST"])
def mentalHealth():

    file = "mental_fitness.xlsx"
    df = pd.read_excel(file)

    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(X[:, 1:])
    X[:, 1:] = imputer.transform(X[:, 1:])

    columnTransformer = ColumnTransformer(
        [('encoder', OneHotEncoder(), [0])],     remainder='passthrough')

    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    knn = KNeighborsClassifier()
    xgb = xgboost.XGBClassifier()
    clf = [('dtc', dtc), ('rfc', rfc), ('knn', knn),
           ('xgb', xgb)]  # list of (str, estimator)

    lr = LogisticRegression()
    stack_model = StackingClassifier(estimators=clf, final_estimator=lr)
    pipe = make_pipeline(columnTransformer, stack_model)

    pipe.fit(X,Y)    

    player_names = [[x for x in request.form.values()]]

    my_prediction = pipe.predict(player_names)

    if my_prediction > 0:
        return render_template('mental.html', pred="Cheers!!!You have shown that you have great fortitude, congrats you are fit to play!")
    else:
        str = Markup("Take your time healing, as long as you want. Sorry!!! You are not fit to play. <br> Here are some valuable tips for your betterment: <br> 1. Spot small opportunities to rest your mind. <br> 2. Give yourself permission to relax. <br> 3. Pay attention to your body as well as your mind. <br> 4. Go to therapy")
        return render_template('mental.html', pred=str)

@app.route("/mental", methods=["GET", "POST"])
def mental():
    return render_template("mental.html")


if __name__ == '__main__':
    app.run(debug=True)
