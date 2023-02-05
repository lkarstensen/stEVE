import numpy as np


def save_line_mesh(point_cloud: np.ndarray, file: str):
    with open(file, "w", encoding="utf-8") as f:
        vertices = [
            f"v {point[0]:.4f} {point[1]:.4f} {point[2]:.4f}\n" for point in point_cloud
        ]
        f.writelines(vertices)
        connections = [f"l {i+1} {i+2}\n" for i in range(point_cloud.shape[0] - 1)]
        f.writelines(connections)
