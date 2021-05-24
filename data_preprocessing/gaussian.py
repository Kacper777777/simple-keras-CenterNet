import numpy as np


def draw_gaussian(heatmap, center, radius, sigma):
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    gaussian = gaussian_2D(r=radius, sigma=sigma)

    left, right = min(x, radius), min(width-x, radius+1)
    top, bottom = min(y, radius), min(height-y, radius+1)

    masked_heatmap = heatmap[y-top:y+bottom, x-left:x+right]
    masked_gaussian = gaussian[radius-top:radius+bottom, radius-left:radius+right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

    return heatmap


def gaussian_2D(r, sigma=1.0):
    y, x = np.ogrid[-r:r+1, -r:r+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size, min_overlap=0.3):
    height, width = det_size
    area = min_overlap * height * width
    smaller_dim = height if height < width else width
    r = area / smaller_dim
    if r < smaller_dim:
        r = np.sqrt(area)
    return r


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    heatmap = np.zeros((10, 14), dtype=np.float32)
    center = (5, 2)
    r = int(gaussian_radius((7, 11), 0.4))
    sigma = max(r/2, 0.01)
    print('center', center, 'r', r, 'sigma', sigma)
    draw_gaussian(heatmap=heatmap, center=center, radius=r, sigma=sigma)
    print(heatmap)
