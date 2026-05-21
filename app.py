from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # ✅ FIX
import matplotlib.pyplot as plt
import os



app = Flask(__name__)
app.config['SECRET_KEY'] = 'pronet_secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
import sqlite3

def create_market_table():
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS market (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        energy REAL,
        price REAL
    )
    """)

    conn.commit()
    conn.close()
# =========================================================
# DATABASE MODELS
# =========================================================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    role = db.Column(db.String(50))
    energy_balance = db.Column(db.Float, default=0.0)

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    seller_id = db.Column(db.Integer)
    buyer_id = db.Column(db.Integer)
    amount = db.Column(db.Float)
    price = db.Column(db.Float)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# =========================================================
# LOAD MODELS
# =========================================================

consumption_model = tf.keras.models.load_model('models/consumption_gru_model_96.h5')
production_model = joblib.load('models/solar_gradient_boosting_model.pkl')

scaler_X = joblib.load("models/consumption_scaler_X.pkl")
scaler_y = joblib.load("models/consumption_scaler_y.pkl")

# =========================================================
# GRAPH FUNCTION
# =========================================================

def plot_graph(consumption, production):

    # Convert 96 → 24
    cons_hourly = np.array(consumption).reshape(24, 4).mean(axis=1)
    prod_hourly = np.array(production).reshape(24, 4).mean(axis=1)

    plt.figure(figsize=(10,5))
    plt.plot(cons_hourly, label="Consumption")
    plt.plot(prod_hourly, label="Production")
    plt.legend()
    plt.title("24 Hour Energy Forecast")

    path = "static/graph.png"
    plt.savefig(path)
    plt.close()

    return path

# =========================================================
# ROUTES
# =========================================================

@app.route('/')
def home():
    return redirect(url_for('login'))

# ------------------ REGISTER ------------------

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':

        if User.query.filter_by(email=request.form['email']).first():
            flash("Email already exists")
            return redirect(url_for('register'))

        user = User(
            username=request.form['username'],
            email=request.form['email'],
            password=request.form['password']
        )

        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html')

# ------------------ LOGIN ------------------

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':

        user = User.query.filter_by(email=request.form['email']).first()

        if user and user.password == request.form['password']:
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid email or password")

    return render_template('login.html')

# ------------------ DASHBOARD ------------------

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

# =========================================================
# INPUT & PREDICTION
# =========================================================

@app.route('/input', methods=['GET', 'POST'])
@login_required
def input_data():

    if request.method == 'POST':

        file = request.files.get('file')

        if not file:
            return "❌ Upload CSV"

        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        required_columns = [
            "Production (W)",
            "From Grid (W)",
            "From Solar (W)",
            "hour",
            "day",
            "month",
            "weekday",
            "day_night",
            "tariff"
        ]

        if not all(col in df.columns for col in required_columns):
            return "❌ Invalid CSV format"

        if len(df) != 96:
            return "❌ Must be 96 rows"

        # ---------------- CONSUMPTION ----------------

        X_scaled = scaler_X.transform(df[required_columns])
        X_input = X_scaled.reshape(1, 96, 9)

        pred_scaled = consumption_model.predict(X_input)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1,1))

        consumption_curve = pred.flatten().tolist()
        next_day_consumption = np.sum(consumption_curve) * 0.25 / 1000

        # ---------------- PRODUCTION ----------------

        df["solar_lag_1"] = df["Production (W)"].shift(1)
        df["solar_lag_2"] = df["Production (W)"].shift(2)
        df["solar_lag_96"] = df["Production (W)"].iloc[0]

        df.fillna(method='bfill', inplace=True)

        solar_features = [
            "hour",
            "month",
            "weekday",
            "day_night",
            "solar_lag_1",
            "solar_lag_2",
            "solar_lag_96"
        ]

        solar_pred = production_model.predict(df[solar_features])
        production_curve = solar_pred.tolist()
        next_day_production = np.sum(production_curve) * 0.25 / 1000

        # ---------------- ROLE ----------------

        net = next_day_production - next_day_consumption
        role = "Producer" if net > 0 else "Consumer"

        current_user.role = role
        current_user.energy_balance = net
        db.session.commit()

        # ---------------- GRAPH ----------------

        graph = plot_graph(consumption_curve, production_curve)

        return render_template(
            "prediction.html",
            consumption=round(next_day_consumption, 2),
            production=round(next_day_production, 2),
            role=role,
            balance=round(net, 2),
            graph=graph
        )

    return render_template("input.html")

# =========================================================
# P2P ENERGY TRADING
# =========================================================

@app.route('/p2p', methods=['GET', 'POST'])
@login_required
def p2p():

    users = User.query.filter(User.id != current_user.id).all()

    if request.method == 'POST':

        buyer = User.query.get(int(request.form['buyer']))
        amount = float(request.form['amount'])
        price = float(request.form['price'])

        if current_user.energy_balance >= amount:

            current_user.energy_balance -= amount
            buyer.energy_balance += amount

            transaction = Transaction(
                seller_id=current_user.id,
                buyer_id=buyer.id,
                amount=amount,
                price=price
            )

            db.session.add(transaction)
            db.session.commit()

            flash("Transaction Successful")
        else:
            flash("Insufficient Balance")

    return render_template('p2p.html', users=users)

# ------------------ LOGOUT ------------------

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# =========================================================

if __name__ == '__main__':
    create_market_table()
    with app.app_context():
        db.create_all()
    app.run(debug=True)