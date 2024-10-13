"""
In this homework we finite discrete method (FDM) to solve the 2D Laplace equations
"""
import numpy as np
import matplotlib.pyplot as plt


def generate_mesh_grid(x_num, y_num):
    """
    param x_num: number of mesh on x-axis
    param y_num: number of mesh on x-axis
    return: x_mesh, y_mesh
    """
    x = np.linspace(0, x_num, x_num+1)
    y = np.linspace(0, y_num, y_num+1)
    x_mesh, y_mesh = np.meshgrid(x, y)
    return x_mesh, y_mesh


def update_algorithm(x_mesh, y_mesh, epsilon=1e-3):
    """
    Utilizing
    param x_mesh:
    param y_mesh:
    param epsilon:
    return: final_field
    """
    phi_list = []
    phi_temp = np.zeros((x_mesh.shape[0], y_mesh.shape[1]))

    phi_temp[100, :] = 100
    phi_temp[:, 100] = 0
    phi_temp[:, 0] = 0
    phi_temp[0, :] = 0
    epsilon_tem = 1
    flag = 0

    while epsilon_tem > epsilon:
        flag += 1
        phi = phi_temp.copy()
        for i in range(1, 100):
            for j in range(1, 100):
                phi_temp[i, j] = 0.25*(phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1])

        epsilon_tem = np.max(np.abs(phi - phi_temp))
        print(flag)

    return phi_temp


if __name__ == '__main__':
    epsilon = 1e-6
    x_mesh, y_mesh = generate_mesh_grid(100, 100)
    phi_temp = update_algorithm(x_mesh, y_mesh, epsilon=epsilon)
    phi_temp = phi_temp[::-1]
    plt.imshow(phi_temp, cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.show()
    print(x_mesh.shape)