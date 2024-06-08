import pandas as pd
import numpy as np


def calculate_dists(data: pd.DataFrame) -> np.array:
    points_count, columns = data.shape
    dists = np.zeros((points_count, points_count), dtype=float)
    for i in range(0, points_count):
        for j in range(i + 1, points_count):
            i_coords = data.iloc[i]
            j_coords = data.iloc[j]
            rad = 0
            for x in range(1, columns):
                rad += (i_coords.iloc[x] - j_coords.iloc[x]) ** 2
            rad = rad ** 0.5
            dists[i, j] = dists[j, i] = rad
    return dists


def calc_E(old: np.array, new: np.array) -> float:
    c = old.sum() / 2
    print(c)
    rows, columns = old.shape
    print(old, new)
    sum = 0
    for i in range(0, rows):
        for j in range(i + 1, columns):
            sum = (((old[i, j] - new[i, j]) ** 2) / old[i, j])

    return sum / c


if __name__ == '__main__':
    IRIS_DATASET = "data/Iris.csv"
    old_points = pd.read_csv(IRIS_DATASET).iloc[:, 0:5]
    print(old_points.iloc[0, 1])
    old_points.loc[-1] = [0, 0, 0, 0, 0]
    old_points.index = old_points.index + 1

    new_points = pd.DataFrame({'x': [0.0], 'y': [0.0]})
    new_points = pd.DataFrame(np.repeat(new_points.values, len(old_points), axis=0), columns=new_points.columns).astype(
        new_points.dtypes)
    new_points['index'] = np.arange(new_points.shape[0])
    new_points = new_points[['index', 'x', 'y']]
    old_dists = calculate_dists(old_points)
    new_dists = calculate_dists(new_points)
    print(new_dists)
    print(calc_E(old_dists, new_dists)*1000)
