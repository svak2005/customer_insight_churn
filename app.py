from flask import Flask, render_template, request
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')  # ✅ Fix: use non-GUI backend for plot saving
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load pre-trained models
with open('models/churn_model.pkl', 'rb') as f:
    churn_model = pickle.load(f)

with open('models/cluster_model.pkl', 'rb') as f:
    cluster_model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    file = request.files['file']
    df = pd.read_csv(file)

    # Match feature names used during model training
    required_columns = ['Age', 'SpendingScore', 'AnnualIncome']
    df = df[required_columns]

    clusters = cluster_model.predict(df)
    df['Cluster'] = clusters

    plt.figure()
    plt.scatter(df['Age'], df['SpendingScore'], c=clusters, cmap='viridis')
    plt.xlabel('Age')
    plt.ylabel('SpendingScore')
    plt.title('Customer Segmentation')
    plot_path = 'static/cluster_plot.png'
    plt.savefig(plot_path)
    plt.close()

    return render_template('results.html', table=df.to_html(classes='table'), image=plot_path)

@app.route('/churn', methods=['POST'])
def churn():
    file = request.files['file']
    df = pd.read_csv(file)

    try:
        churn_pred = churn_model.predict(df)
    except ValueError as e:
        return f"❌ Error predicting churn: {e}"

    df['Churn_Prediction'] = churn_pred
    return render_template('results.html', table=df.to_html(classes='table'), image=None)

if __name__ == '__main__':
    app.run(debug=True)
