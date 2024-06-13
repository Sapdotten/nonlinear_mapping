import numpy as np


class Gradient:
    """
    Basic class for find most fittig distances between points
    """
    C: np.array  # old distances, (N, N)
    d: np.array  # new distances (N, N)
    y: np.array  # new coords (N, 2)
    old_sum: float  # sum of old distances to calculate error
    N: int
    MF: float = 0.3

    def __init__(self, points: np.array):
        """Creates class for working with points from np.array

        Args:
            points (np.array): array with points coords.
            each column is yi coord, each row is one point.
        """
        # calculate old distances by points coords
        self.N, _ = points.shape
        self.d = np.zeros(shape=(self.N, self.N))
        self.y = points
        self.calculate_distances()
        self.C = self.d.copy()

        # here should be code for calculate new points
        self.fill_coords_random()
        self.calculate_distances()

        # calculate sum if distances for old points
        self.old_sum = self.C.sum()/2

    def fill_coords_random(self):
        self.y = np.rand((self.N, self.N))

    def fill_coords_PCA(self, old_coords: np.array):
        """Calculates new coords using PCA

        Args:
            old_coords (np.array): coords like in __init__
        """
        rows, columns = old_coords.shape
        centered_coords = []
        # calculating of centered values of coords
        for i in range(0, columns):
            centered_coords.append(old_coords[:, i] - old_coords[:, i].mean())

        # calculating of covariation matrix
        covariation_matrix = np.cov(centered_coords)

        sum_dispersion = np.trace(covariation_matrix)
        first_max_dispersion = None
        first_important_axe = None
        second_max_dispersion = None
        second_important_axe = None

        for i in range(0, len(covariation_matrix)):
            percent = covariation_matrix[i, i]/sum_dispersion
            if first_max_dispersion is None or percent > first_max_dispersion:
                first_max_dipsersion = percent
                first_important_axe = i
            elif second_max_dispersion is None or percent > second_max_dispersion:
                second_max_dispersion = percent
                second_important_axe = i

        values, vectors = np.linalg.eig(covariation_matrix)
        self.y[0] = np.dot(vectors[:, first_important_axe], centered_coords)
        self.y[1] = np.dot(vectors[:, second_important_axe], centered_coords)

    def calculate_error(self) -> float:
        """
        Calculates error of distances
        :return: error * 100
        """
        dist_sum = 0
        for i in range(0, self.N):
            for j in range(i + 1, self.N):
                dist_sum += (((self.C[i, j] - self.d[i, j])
                              ** 2) / self.C[i, j] + 0.0001)

        return dist_sum / (self.old_sum + 0.0001)

    def calculate_distances(self):
        """
        Calculates distances for self.y 
        """
        rows, columns = self.y.shape
        for i in range(0, rows):
            for j in range(i + 1, rows):
                axe_dist = 0
                for y in range(0, columns):
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
