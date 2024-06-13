import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class Points:
    coords: np.array
    distances: np.array

    def __init__(self, point_count: int, coords_count: int):
        self.coords = np.random.rand(point_count, coords_count)
        self.distances = np.zeros(shape=(point_count, point_count))

    def from_numpy(self, array: np.array):
        self.coords = array
        self.distances = np.zeros(shape=(len(array), len(array)))
        self.calculate_distances()
        self.coords[np.isnan(self.coords)] = 0
        self.distances[np.isnan(self.distances)] = 0

    def calculate_distances(self):
        rows, columns = self.coords.shape
        for i in range(0, rows):
            for j in range(i + 1, rows):
                axe_dist = 0
                for y in range(0, columns):
                    axe_dist += (self.coords[i, y] - self.coords[j, y]) ** 2
                self.distances[i, j] = self.distances[j, i] = axe_dist ** 0.5


class Gradient:
    C: np.array  # old distances, (N, N)
    d: np.array  # new distances (N, N)
    y: np.array  # new coords (N, 2)
    old_sum: float  # sum of old distances to calculate error
    N: int
    MF: float = 0.3

    def __init__(self, old_points: Points):
        self.C = old_points.distances
        self.N, _ = self.C.shape
        self.y = np.random.uniform(0, 1, size=(self.N, self.N))
        self.d = np.ones(shape=(self.N, self.N))

        self.calculate_distances()
        self.d[np.isnan(self.d)] = 0
        self.y[np.isnan(self.y)] = 0
        self.old_sum = self.C.sum() / 2

    def calculate_error(self) -> float:
        """
        Calculates error of distances
        :return: error
        """
        dist_sum = 0
        for i in range(0, self.N):
            for j in range(i + 1, self.N):
                dist_sum += (((self.C[i, j] - self.d[i, j])
                             ** 2) / (self.C[i, j]+0.01))
        return dist_sum / (self.old_sum)

    def calculate_distances(self):
        for i in range(0, self.N):
            for j in range(i + 1, self.N):
                axe_dist = 0
                for y in range(0, 2):
                    axe_dist += (self.y[i, y] - self.y[j, y]) ** 2
                self.d[i, j] = self.d[j, i] = axe_dist ** 0.5

    def delta(self, p: int, q: int) -> float:
        """
        Calculates delta for one coord of one point
        :param p: number of point
        :param q: number of y-coord
        :return: calculated delta
        """
        result = 0.0
        denominator = 0
        numerator = 0
        for j in range(0, self.N):
            if j == p:
                continue
            else:
                numerator += (((self.C[p, j] - self.d[p, j]) / (self.d[p, j] * self.C[p, j] + 0.001)) * (
                    self.y[p, q] - self.y[j, q]))
                denominator += (1 / (self.C[p, j] * self.d[p, j] + 0.001))

                #
                # denominator = (
                #     abs((-2) * ((self.d[n, p] ** 2) * self.C[p, n]) + (2 * self.y[p, q]) * (
                #             self.y[p, q] + self.y[n, q]) * (self.C[p, n]) - 4 * (self.d[n, p] ** 3)))
                # if denominator == 0:
                #     denominator = 0.01
                # numerator = (((self.C[p, n] * self.d[n, p]) - 2 * (self.y[p, q] - self.y[n, q]) * (
                #         self.C[p, n] + 2 * self.d[n, p])) * (self.d[n, p]) ** 2)
                # result += numerator / denominator
        result = -numerator / (abs(denominator) + 0.001)
        return result

    def new_coords(self):
        new_y = self.y.copy()
        for p in range(0, self.N):
            for q in range(0, 2):
                new_y[p, q] = self.y[p, q] - self.MF * self.delta(p, q)
        self.y = new_y
        self.calculate_distances()


if __name__ == '__main__':
    points = Points(10, 2)
    IRIS_DATASET = "data/Iris.csv"
    DISTANCES = "data/distances.csv"

    data = pd.read_csv(IRIS_DATASET).iloc[:, 1:5]
    points.from_numpy(data.to_numpy())
    grad = Gradient(points)
    # plt.scatter(grad.y[0], grad.y[1])
    # plt.show()
    # for i in range(0, 100):
    #
    #     print(f"Error is {grad.calculate_error()}")
    #     grad.new_coords()
    # plt.scatter(grad.y[0], grad.y[1])
    # plt.show()
    rows, columns = points.coords.shape
    X_centered = []
    for i in range(0, columns):
        # points.coords[:, i] = points.coords[:, i] - points.coords[:, i].mean()
        X_centered.append(points.coords[:, i])
    print(points.coords.mean())
    covmat = np.cov(X_centered)
    print(covmat)
    sum_disp = 0
    for i in range(0, len(covmat)):
        sum_disp += covmat[i, i]
    print("Проценты сохранения информации:")
    for i in range(0, len(covmat)):
        print(f"Ось {i}: {covmat[i, i]/sum_disp}")
    values, vectors = np.linalg.eig(covmat)
    print(values)
    print(vectors)
    v = (vectors[:, 0])
    print(f"V shape is {v.shape}")
    print("resize:")
    Xnew = np.dot(v, X_centered)
    print(Xnew.shape)
    Ynew = np.dot(vectors[:, 2], X_centered)
    # plt.subplot(1, 3, 1)
    # plt.scatter(Xnew, Ynew)
    # grad.y[:, 1] = Ynew
    # grad.y[:, 0] = Xnew
    plt.subplot(1, 3, 1)
    plt.scatter(grad.y[:, 0], grad.y[:, 1])
    grad.calculate_distances()
    plt.subplot(1, 3, 3)
    errors = []
    for i in range(0, 700):
        error = grad.calculate_error()
        errors.append(error)
        print(f"{i}) Error is {error}")
        grad.new_coords()
    plt.plot(range(0, 700), errors)
    plt.subplot(1, 3, 2)
    plt.scatter(grad.y[:, 0], grad.y[:, 1])

    plt.show()
