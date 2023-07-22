import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.cluster import KMeans
import math
import folium
import base64
from io import BytesIO
import zipfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os

# Function to check if a point is within a boundary box
def is_within_boundary(point, boundary_box):
    x, y = point
    x1, x2, y1, y2 = boundary_box
    return x1 <= x <= x2 and y1 <= y <= y2

def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def find_nearest_point(reference_point, points_set):
    if not points_set:
        raise ValueError("The points_set cannot be empty.")

    nearest_point = points_set[0]
    min_distance = euclidean_distance(reference_point, nearest_point)

    for point in points_set[1:]:
        distance = euclidean_distance(reference_point, point)
        if distance < min_distance:
            nearest_point = point
            min_distance = distance

    return nearest_point


def load_data(csv):
    df = pd.read_csv(csv)
    df = df.rename(columns={"Geo - Longitude" : "long", "Geo - Latitude": "lat"})
    df = df[['Premises ID', 'lat', 'long']]

    # drop duplicates
    df['latlong'] = df['lat'].astype(str) + df['long'].astype(str)
    df = df.drop_duplicates(subset='latlong', keep=False)

    return df


def generate_scatterplot_1(df):
    sns.scatterplot(x=df['lat'], y=df['long'])

    plt.xlabel('Latitude', fontsize=8)
    plt.ylabel('Longitude', fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=6)

    # create an image and somehow return this so that it can be displayed in the html

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_image = base64.b64encode(buffer.read()).decode()

    buffer.flush()
    buffer.close()

    return plot_image

def KMeansAlgo(df, PATH_DISTANCE=5, NUM_SHIFTS=10):
    # Assuming 'coordinates' is a 2D list of longitude and latitude pairs
    # Example: coordinates = [[long1, lat1], [long2, lat2], ...]

    # Convert the coordinates list into a numpy array
    data = df[['lat', 'long']]

    # Define the number of centroids you want (9 in this case)
    k = NUM_SHIFTS

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data)

    # Get the cluster centroids
    centroids = kmeans.cluster_centers_

    # 'centroids' now contains the 9 centroids representing your clusters
    c = pd.DataFrame(centroids, columns=['lat', 'long'])
    return c

def generate_scatterplot_2(df, c):
    sns.scatterplot(x=df['lat'], y=df['long'])
    sns.scatterplot(x=c['lat'], y=c['long'], color='red')

    plt.xlabel('Latitude', fontsize=8)
    plt.ylabel('Longitude', fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=6)

    # create an image and somehow return this so that it can be displayed in the html
    buffer = BytesIO()
    plt.savefig(buffer, format='png')

    buffer.seek(0)
    plot_image = base64.b64encode(buffer.read()).decode()

    buffer.flush()
    buffer.close()

    return plot_image

def findBoundingBoxes(df, c):
    # get range of data
    x_min = np.min(df['lat'])
    x_max = np.max(df['lat'])
    y_min= np.min(df['long'])
    y_max= np.max(df['long'])

    # get bounding box 10% of addresses around each k means centroid
    x_diff = x_max - x_min
    y_diff = y_max - y_min
    x_delta = 0.1 * x_diff
    y_delta = 0.1 * y_diff

    c['x1'] = c['lat'] - x_delta
    c['x2'] = c['lat'] + x_delta

    c['y1'] = c['long'] - y_delta
    c['y2'] = c['long'] + y_delta

    return c

def generate_scatterplot_3(df, c):
    # Create a scatter plot using Seaborn
    sns.scatterplot(x=df['lat'], y=df['long'])
    sns.scatterplot(x=c['lat'], y=c['long'])

    # Loop through each row and add the square for each set of coordinates
    for _, row in c.iterrows():
        x1, x2, y1, y2 = row['x1'], row['x2'], row['y1'], row['y2']
        square_side_length = abs(x2 - x1)
        square = Rectangle((x1, y1), square_side_length, square_side_length, linewidth=1, edgecolor='red', facecolor='none')
        plt.gca().add_patch(square)

    # # Set axis labels and show the plot
    plt.xlabel('Latitude', fontsize=8)
    plt.ylabel('Longitude', fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=6)
    # plt.title('Squares for Each Row')

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_image = base64.b64encode(buffer.read()).decode()

    buffer.flush()
    buffer.close()

    return plot_image



def shortestPath(df, centroids, PATH_DISTANCE=5, NUM_SHIFTS=10):
    points_list = []

    # for each boundary box
    for _, boundary_box in centroids.iterrows():
        points_within_boundary = df[df.apply(lambda row: is_within_boundary((row['lat'], row['long']), boundary_box[['x1', 'x2', 'y1', 'y2']]), axis=1)]
        points_list.append(points_within_boundary)


    results = {}

    print("len of centroids", len(centroids))

    # loop over bounding boxes
    for bb in range(len(centroids)):
        print(f"Finding shortest path in bb {bb}")

        points = points_list[bb]

        path_best = []
        min_distance = 100000000

        for _, point in points.iterrows():
            path = []
            distance = 0

            # starting point
            reference_point = (point['lat'], point['long'])

            # filter point from point set
            points_set = list(zip(df['lat'], df['long']))

            for step in range(PATH_DISTANCE):

                points_set.remove(reference_point)

                # find nearest point to current point
                nearest = find_nearest_point(reference_point, points_set)
                distance += euclidean_distance(reference_point, nearest)

                # print(reference_point, "-", nearest, "distance=", distance)

                reference_point = nearest
                path.append(reference_point)

            if distance < min_distance:
                min_distance = distance
                path_best = path

        results[bb] = {"distance": min_distance, "path": path_best}

    return results


def generateMap(df, results):

    # Create a folium map centered on South Africa
    map_center = [df['lat'].mean(), df['long'].mean()]
    zoom_level = 14
    map_osm = folium.Map(location=map_center, zoom_start=zoom_level)


    for k in list(results.keys()):
        path = results[k]['path']
        lats, longs = [x[0] for x in path], [x[1] for x in path]

        # Add points to the map
        for lat, lon in zip(lats, longs):
            folium.Marker([lat, lon], popup=f'Lat: {lat}, Lon: {lon}').add_to(map_osm)

        # Create a list of coordinates for the line
        line_coordinates = list(zip(lats, longs))

        # Add the line to the map
        folium.PolyLine(locations=line_coordinates, color='blue').add_to(map_osm)

    # Display the map
    map_osm.save('templates/iframe.html')


def create_pdf_from_string(file_name, text_string):
    # Create a PDF document
    doc = SimpleDocTemplate(file_name, pagesize=letter)
    styles = getSampleStyleSheet()

    # Create a list to hold the story (content) elements
    story = []

    # Set the style for the text
    text_style = styles["Normal"]

    # Add the string to the story with automatic line wrapping
    p = Paragraph(text_string, text_style)
    story.append(p)

    # Build the PDF
    doc.build(story)


# clear pdfs folder
def delete_all_files_in_folder(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Loop through the files and delete each one
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def generatePdfs(centroids, results, df, PATH_DISTANCE=5, NUM_SHIFTS=10):

    # os.mkdir("pdfs/")
    # generate pdfs
    for bb in range(len(centroids)):
        file_name = f"pdfs/shift_{bb}.pdf"
        # generate text for pdf
        text = f"Route with {PATH_DISTANCE} stops for shift {bb+1} <br/><br/>"
        for i, stop in enumerate(results[bb]['path']):
            address = list(df[(df['lat'] == stop[0]) & (df['long'] == stop[1])]['Premises ID'])[0]
            text += f"<br/>{i+1}, {stop}, {address}"
        # save pdf
        create_pdf_from_string(file_name, text)


def zip_folder(folder_path, zip_path):
    # Create a ZIP file
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Loop through all the files in the folder and add them to the ZIP
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, folder_path)
                zip_file.write(file_path, relative_path)
