from flask import Flask
from flask import render_template, request
from function import artist_data, get_recommandation, recommandation_for_you, for_you, for_you_sim
import pandas as pd
import numpy as np

app = Flask(__name__)




@app.route('/', methods=['GET','POST'])
def index():
    artist = artist_data()
    if request.method == 'POST':
        #print(request.form.getlist("artistcheckbox"))
        return 'Done'
    return render_template('index.html', artist = artist)

@app.route('/page1', methods=['GET','POST'])
def artist_recommande():
    request.method == 'POST'

    artist_choice = request.form.getlist("artistcheckbox")
    recommandation = get_recommandation(artist_choice)
    recom_for_you  = recommandation_for_you(artist_choice)
    For_you = for_you(artist_choice)
    For_you_sim = for_you_sim(artist_choice)

    return render_template("page.html", recommandation= recommandation, recom_for_you = recom_for_you
                    , For_you=For_you, For_you_sim = For_you_sim)

