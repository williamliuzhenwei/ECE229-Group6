from flask import Flask

app = Flask(__name__)
@app.route("/")
def homepage():
    return "<h1> ECE229 Group6 </h1>"
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)
