from flask import Flask, render_template, redirect, url_for, flash, request
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import mysql.connector
import warnings
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Database configuration
db = mysql.connector.connect(
    host="localhost",
    port=3308,
    user="root",
    password="root",
    database="social_media"
)
cursor = db.cursor()

app = Flask(__name__)
app.secret_key = 'secret'
warnings.filterwarnings("ignore", category=UserWarning)

# Load pretrained models
random_forest = pickle.load(open('random_fake.pkl', 'rb'))
decision_tree = pickle.load(open('decision_fake.pkl', 'rb'))

# Helper to load test data
def load_test_data(csv_path='C:/Users/sadiy/OneDrive/Desktop/Capstone/PIP -40001/Fake social media profile detection and reporting/model/Dataset/instagram.csv'):
    df = pd.read_csv(csv_path, encoding='unicode_escape')
    X = df.drop(['fake'], axis=1).values
    y = df['fake'].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    return X_test, y_test

# Evaluate and prepare metrics for a model
def load_and_evaluate(model, model_name, X_test, y_test, image_filename):
    # Predict
    y_pred = model.predict(X_test)
    # Compute classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Plot and save confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f'static/{image_filename}')
    plt.close()
    # Build metrics list
    metrics = [
        {'label': '0', 'recall': report['0']['recall'], 'precision': report['0']['precision'], 'f1_score': report['0']['f1-score']},
        {'label': '1', 'recall': report['1']['recall'], 'precision': report['1']['precision'], 'f1_score': report['1']['f1-score']}
    ]
    return {
        'name': model_name,
        'metrics': metrics,
        'image': image_filename
    }

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    email = request.form.get('email')
    password = request.form.get('password')
    cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
    user = cursor.fetchone()
    if user:
        flash("Login successful!", "success")
        return redirect(url_for('upload'))
    else:
        flash("Invalid credentials. Please try again.", "danger")
        return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form.get('fullname')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if password != confirm_password:
            flash("Passwords do not match", "danger")
            return redirect(url_for('signup'))
        try:
            cursor.execute(
                "INSERT INTO users (fullname, email, password) VALUES (%s, %s, %s)",
                (fullname, email, password)
            )
            db.commit()
            flash("Signup successful!", "success")
            return redirect(url_for('login'))
        except Exception as e:
            db.rollback()
            flash(f"Error: {e}", "danger")
    return render_template('signup.html')

@app.route('/upload')
def upload():
    return render_template("upload.html")

@app.route('/preview', methods=["POST"])
def preview():
    dataset = request.files['datasetfile']
    df = pd.read_csv(dataset)
    return render_template("preview.html", df_view=df)

@app.route('/prediction')
def prediction():
    return render_template("prediction.html")

@app.route('/predict', methods=["POST"])
def predict():
    sample_data = [
        request.form['Profile'], request.form['Length_Username'], request.form['Fullname'],
        request.form['Length_Fullname'], request.form['Name==Username'], request.form['Description_Length'],
        request.form['External_URL'], request.form['Account_Private'], request.form['Total_Posts'],
        request.form['Total_Followers'], request.form['Total_Follows']
    ]
    features = np.array([float(i) for i in sample_data]).reshape(1, -1)
    model_choice = request.form['model']
    if model_choice == 'RandomForestClassifier':
        pred = random_forest.predict(features)[0]
    else:
        pred = decision_tree.predict(features)[0]
    result = 'Fake' if pred == 1 else 'Real'
    return render_template('prediction.html', prediction_text=result, model=model_choice)

@app.route('/performance')
def performance():
    # Prepare test data
    X_test, y_test = load_test_data()
    # Evaluate both models
    results = []
    results.append(load_and_evaluate(random_forest, 'RandomForestClassifier', X_test, y_test, 'random_confusion.png'))
    results.append(load_and_evaluate(decision_tree, 'DecisionTreeClassifier', X_test, y_test, 'decision_confusion.png'))
    return render_template('performance.html', model_results=results)

@app.route('/chart')
def chart():
    models = [
        {'name': 'RandomForestClassifier', 'accuracy': 93},
        {'name': 'DecisionTreeClassifier', 'accuracy': 92}
    ]

    class_distribution = [
        {'class': 'Fake', 'percentage': 60},
        {'class': 'Real', 'percentage': 40}
    ]

    return render_template('chart.html', models=models, class_distribution=class_distribution)


if __name__ == '__main__':
    app.run(debug=True)
