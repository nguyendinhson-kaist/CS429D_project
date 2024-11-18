import plotly.graph_objects as go
import numpy as np
import torch
import torch.nn as nn


def box_blur(voxel_grid, threshold:int=1):
    ksize = 3
    kvol = ksize ** 3
    threshold = threshold / kvol - 1e-6 

    if isinstance(voxel_grid, np.ndarray):
        voxel_grid = torch.from_numpy(voxel_grid).float()
    conv = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
    # Initialize weight of conv to make it as a mean filter
    conv.weight.data.fill_(1.0/27)
    voxel_grid = voxel_grid.unsqueeze(0).unsqueeze(0)
    voxel_grid = conv(voxel_grid)
    voxel_grid = (voxel_grid > threshold).float()
    return voxel_grid.squeeze(0).squeeze(0).numpy()


def get_voxel_plotly(voxel_grid, lim=128):
    """
    Visualizes a 3D binary voxel grid using Plotly.

    Parameters:
    voxel_grid (numpy.ndarray): A 3D binary voxel grid where 1 indicates occupancy and 0 indicates empty.
    """

    # change to numpy if needed
    if isinstance(voxel_grid, torch.Tensor):
        voxel_grid = voxel_grid.cpu().numpy()

    # Get the coordinates of occupied voxels
    occupied_voxels = np.argwhere(voxel_grid == 1)

    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=occupied_voxels[:, 0],
        y=occupied_voxels[:, 2],
        z=occupied_voxels[:, 1],
        mode='markers',
        marker=dict(
            size=2,
            color='blue',
        )
    )])

    # Set labels
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            xaxis=dict(range=[0, lim]),
            yaxis=dict(range=[0, lim]),
            zaxis=dict(range=[0, lim])
        )
    )

    return fig
