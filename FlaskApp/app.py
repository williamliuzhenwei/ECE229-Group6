from flask import Flask, render_template, request, redirect, url_for
from model import custom_recommender, item_based_recommender, content_based_recommender#, custom_recommender
from model_1 import get_book_title
import requests
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

books = pd.read_csv('new_data.csv')
df = books.copy()

app = Flask(__name__)
auth = HTTPBasicAuth()

users = {
    "dev": generate_password_hash("dev1"),
    "test": generate_password_hash("test1")
}

@auth.verify_password
def verify_password(username, password):
    if username in users and \
            check_password_hash(users.get(username), password):
        return username

@app.route("/", methods=['GET'])
def homepage():

    return render_template('ECEGroup6.html')

@app.route("/index", methods=['GET'])
def index():
    return render_template('IndexError.html')

@app.route("/other", methods=['GET'])
def other():
    return render_template('OtherError.html')

@app.route('/search', methods=['GET', 'POST'])
@auth.login_required
def form_example():
    # handle the POST request
    result = 'none yet'
    if request.method == 'POST':
        book = request.form.get('book')
        model = request.form.get('select_model')
        if model == 'Cosine_sim':
            #df = clean_df(df)
            #tfidf_vectors = get_tfidf_vectors(df)
            #result = recommendations(book, tfidf_vectors)
            try:
                title,img = get_book_title(book)
            except IndexError:
                return redirect('/index')
            img1 = img[0]
            img2 = img[1]
            img3 = img[2]
            img4 = img[3]
            img5 = img[4]
            title1 = title[0]
            title2 = title[1]
            title3 = title[2]
            title4 = title[3]
            title5 = title[4]
            try:
                return render_template('res.html', book=book, model=model, title1=title1, title2=title2,title3=title3,title4=title4,title5=title5, \
         img1=img1,img2 = img2, img3=img3,img4=img4,img5=img5)
            except:
                return redirect('other')
            
        elif model == 'Item-based':
             
            recom_data = item_based_recommender(book)
            
            if type(recom_data) != pd.DataFrame and recom_data == None:
                return redirect('/index')
            links = []
            titles = []
            for i in range(len(recom_data['book_title'].tolist())):
                url = books.loc[books['book_title'] == recom_data['book_title'].tolist()[i],'img_l'][:1].values[0]
                title = books.loc[books['book_title'] == recom_data['book_title'].tolist()[i],'book_title'][:1].values[0]
                links.append(url)
                titles.append(title) 
            
            img1 = links[0]
            img2 = links[1]
            img3 = links[2]
            img4 = links[3]
            img5 = links[4]

            title1 = titles[0]
            title2 = titles[1]
            title3 = titles[2]
            title4 = titles[3]
            title5 = titles[4]
            try:
                return render_template('res.html', book=book, model=model, title1=title1, title2=title2,title3=title3,title4=title4,title5=title5, \
         img1=img1,img2 = img2, img3=img3,img4=img4,img5=img5)
            except:
                return redirect('/other')
            
            # return render_template('res.html', book=book, model=model, fig='static/assets/img/res.png')

        elif model == 'Content-based':
            try:
                title,img = content_based_recommender(book)
            except TypeError:
                return redirect('/index')
            img1 = img[0]
            img2 = img[1]
            img3 = img[2]
            img4 = img[3]
            img5 = img[4]
            title1 = title[0]
            title2 = title[1]
            title3 = title[2]
            title4 = title[3]
            title5 = title[4]
            try:
                return render_template('res.html', book=book, model=model, title1=title1, title2=title2,title3=title3,title4=title4,title5=title5, \
         img1=img1,img2 = img2, img3=img3,img4=img4,img5=img5)
            except:
                return redirect('/other')
            return render_template('res.html', book=book, model=model, fig='static/assets/img/res.png')
        else:
            try:
                title,img = custom_recommender(book)
            except TypeError:
                return redirect('/index')
            img1 = img[0]
            img2 = img[1]
            img3 = img[2]
            img4 = img[3]
            img5 = img[4]
            title1 = title[0]
            title2 = title[1]
            title3 = title[2]
            title4 = title[3]
            title5 = title[4]
            try:
                return render_template('res.html', book=book, model=model, title1=title1, title2=title2,title3=title3,title4=title4,title5=title5, \
            img1=img1,img2 = img2, img3=img3,img4=img4,img5=img5)
            except:
                return redirect('/other')
            # custom_recommender(book)

        #framework = request.form.get('framework')
        

    # otherwise handle the GET request
    return render_template('recommend.html')


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0',port=8080)
    #app.run()
