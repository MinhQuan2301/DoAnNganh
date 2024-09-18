from urllib.parse import quote

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
import matplotlib
from sklearn.linear_model import LogisticRegression

app = Flask(__name__, template_folder='templates')
app.secret_key = "1234567890!@#$%^&*()qwertyuioplkjhgfdsazxcvbnm,./ASDFGHJKLZMXNCBVQWERTYUIOP"
app.config["SQLALCHEMY_DATABASE_URI"] = ("mysql+pymysql://root:%s@localhost/library?charset=utf8mb4"
                                         % quote("d@Ikaquan2301"))
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True

db = SQLAlchemy(app=app)

admin = Admin(app, name='Kho Dữ Liệu', template_mode='bootstrap4')
matplotlib.use('agg')
model = LogisticRegression()
