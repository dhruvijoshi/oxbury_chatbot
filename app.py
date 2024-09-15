from flask import Flask, render_template, request, jsonify


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('main.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    pass


if __name__ == '__main__':
    app.run()