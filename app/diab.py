from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

from flask import request

@app.route('/evaluate', methods=['POST'])
def evaluate():
    if request.method == 'POST':
        # load model
        return do_the_login()
    else:
        return show_the_login_form()

@app.route('/evaluate', methods=['POST'])
def results(ans):
    return "uwu"