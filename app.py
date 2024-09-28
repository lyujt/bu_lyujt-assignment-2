from flask import Flask, jsonify, request, render_template
import numpy as np

app = Flask(__name__)


class KMeans:
    def __init__(self, k, init_method="random"):
        self.k = k
        self.init_method = init_method
        self.centroids = None

    def initialize_centroids(self, data):
        if self.centroids is not None:
            return
        if self.init_method == "random":
            self.centroids = data[np.random.choice(data.shape[0], self.k, replace=False)]
        elif self.init_method == "farthest":
            self.centroids = self.farthest_first_initialization(data)
        elif self.init_method == "kmeans++":
            self.centroids = self.kmeans_plus_plus(data)

    def farthest_first_initialization(self, data):
        centroids = [data[np.random.choice(data.shape[0])]]
        for _ in range(1, self.k):
            distances = np.array([np.min([np.linalg.norm(x - c) for c in centroids]) for x in data])
            centroids.append(data[np.argmax(distances)])
        return np.array(centroids)

    def kmeans_plus_plus(self, data):
        centroids = [data[np.random.choice(data.shape[0])]]
        for _ in range(1, self.k):
            distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids]) for x in data])
            probabilities = distances / np.sum(distances)
            centroids.append(data[np.random.choice(data.shape[0], p=probabilities)])
        return np.array(centroids)

    def assign_clusters(self, data):
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def update_centroids(self, data, labels):
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.k)])
        return new_centroids

    def step(self, data):
        """Performs one iteration of the KMeans algorithm."""
        labels = self.assign_clusters(data)
        new_centroids = self.update_centroids(data, labels)
        return new_centroids, labels

    def fit(self, data, max_iters=100, tol=1e-4):
        """Runs the KMeans algorithm until convergence."""
        self.initialize_centroids(data)
        for iteration in range(max_iters):
            labels = self.assign_clusters(data)
            new_centroids = self.update_centroids(data, labels)
            centroid_shift = np.linalg.norm(self.centroids - new_centroids, axis=1)
            if np.all(centroid_shift < tol):
                return self.centroids, labels, True
            self.centroids = new_centroids
        return self.centroids, labels, False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate_dataset', methods=['POST'])
def generate_dataset():
    data = np.random.rand(100, 2)
    return jsonify(data.tolist())


@app.route('/kmeans_step_by_step', methods=['POST'])
def kmeans_step_by_step():
    content = request.json
    data = np.array(content['data'])
    k = content['k']
    method = content['method']
    iteration = content.get('iteration', 0)

    kmeans = KMeans(k=k, init_method=method)

    # Initialize centroids on the first iteration
    if 'centroids' in content and len(content['centroids']) > 0:
        kmeans.centroids = np.array(content['centroids'])
    elif iteration == 0:
        kmeans.initialize_centroids(data)  # Initialize centroids at the first step

    # Perform one step of KMeans (only one iteration)
    new_centroids, labels = kmeans.step(data)

    # Check if centroids have converged (no change)
    centroid_shift = np.linalg.norm(kmeans.centroids - new_centroids, axis=1)
    converged = bool(np.all(centroid_shift < 1e-4))  # Make sure converged is a proper Python bool

    # Update centroids for next iteration
    kmeans.centroids = new_centroids

    return jsonify({
        'centroids': new_centroids.tolist(),  # Convert NumPy array to list
        'labels': labels.tolist(),  # Convert NumPy array to list
        'converged': converged  # Ensure boolean is serializable
    })


@app.route('/run_to_converge', methods=['POST'])
def run_to_converge():
    content = request.json
    data = np.array(content['data'])
    k = content['k']
    method = content['method']

    kmeans = KMeans(k=k, init_method=method)

    # Initialize centroids if provided manually
    if 'centroids' in content and len(content['centroids']) > 0:
        kmeans.centroids = np.array(content['centroids'])

    # Run KMeans until convergence
    centroids, labels, converged = kmeans.fit(data)

    return jsonify({
        'centroids': centroids.tolist(),
        'labels': labels.tolist(),
        'converged': converged  # Now we check for convergence
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
