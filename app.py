from flask import Flask, request, render_template, jsonify, send_from_directory
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from processes import load_data, generate_scatterplot_1, KMeansAlgo, generate_scatterplot_2, findBoundingBoxes, generate_scatterplot_3, shortestPath, generateMap, zip_folder, delete_all_files_in_folder, generatePdfs
import os

app = Flask(__name__)

# Set the "Agg" backend for Matplotlib
matplotlib.use('Agg')


@app.route("/", methods=["GET"])
def upload_form():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload_csv():
    NUM_SHIFTS = int(request.form.get("NUM_SHIFTS"))
    PATH_DISTANCE = int(request.form.get("PATH_DISTANCE"))

    csv_file = request.files["csv_file"]
    if csv_file.filename.endswith(".csv"):
        # Process the CSV file and get the DataFrame
        df = load_data(csv_file)

        c = KMeansAlgo(df, PATH_DISTANCE=PATH_DISTANCE, NUM_SHIFTS=NUM_SHIFTS)

        c = findBoundingBoxes(df, c)

        # generate images

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(6, 4))
        plot_image1 = generate_scatterplot_1(df)
        plt.close()

        plt.figure(figsize=(6, 4))
        plot_image2 = generate_scatterplot_2(df, c)
        plt.close()


        plt.figure(figsize=(6, 4))
        plot_image3 = generate_scatterplot_3(df, c)
        plt.close()


        # Process the CSV file and generate the scatter plot as an image
        return jsonify({'images': [plot_image1, plot_image2, plot_image3] })
    else:
        return jsonify({'error': 'Please upload a valid CSV file.'})


@app.route("/generateMap", methods=["POST"])
def generateOutputMap():
    NUM_SHIFTS = int(request.form.get("NUM_SHIFTS"))
    PATH_DISTANCE = int(request.form.get("PATH_DISTANCE"))

    csv_file = request.files["csv_file"]
    # Process the CSV file and get the DataFrame
    df = load_data(csv_file)

    c = KMeansAlgo(df, PATH_DISTANCE=PATH_DISTANCE, NUM_SHIFTS=NUM_SHIFTS)

    c = findBoundingBoxes(df, c)

    results = shortestPath(df, c, NUM_SHIFTS=NUM_SHIFTS, PATH_DISTANCE=PATH_DISTANCE)

    generateMap(df, results)

    folder_path = "pdfs/"
    if os.path.exists(folder_path):
        delete_all_files_in_folder(folder_path)

    generatePdfs(c, results, df, PATH_DISTANCE=PATH_DISTANCE, NUM_SHIFTS=NUM_SHIFTS)

    zip_folder("pdfs/", "static/results.zip")

    return jsonify({'success': True })
    # return render_template("iframe.html")

@app.route("/iframe", methods=["GET"])
def iframe():
    return render_template("iframe.html")

@app.route("/download", methods=["GET"])
def download():
    directory = 'static'  # Specify the directory where the files are located
    return send_from_directory(directory, 'results.zip', as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
