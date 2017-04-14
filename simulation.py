import numpy as np

def generate_minibatch(shape, count):
    x, y = zip(*[generate_random_data((shape[1], shape[2])) for i in range(0, count)])

    X = np.reshape(np.hstack(x), (count, 1, shape[1], shape[2]))
    Y = np.reshape(np.hstack(y), (count, 1, shape[1], shape[2]))

    return X.astype(np.float32), Y.astype(np.float32)

def generate_random_data(shape):
    triangle_location = get_random_location(shape)
    circle_location = get_random_location(shape)
    mesh_location = get_random_location(shape)

    # Create input image
    arr = np.zeros(shape, dtype=bool)
    arr = add_triangle(arr, *triangle_location)
    arr = add_circle(arr, *circle_location)
    arr = add_mesh_square(arr, *mesh_location)

    # Create target mask
    mask = np.zeros(shape, dtype=bool)
    mask = add_circle(mask, *circle_location)

    return arr, mask

def add_square(arr, x, y, size):
    s = int(size / 2)
    arr[x-s,y-s:y+s] = True
    arr[x+s,y-s:y+s] = True
    arr[x-s:x+s,y-s] = True
    arr[x-s:x+s,y+s] = True

def add_filled_square(arr, x, y, size):
    s = int(size / 2)

    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

    return np.logical_or(arr, logical_and([xx > x - s, xx < x + s, yy > y - s, yy < y + s]))

def logical_and(arrays):
    new_array = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        new_array = np.logical_and(new_array, a)

    return new_array

def add_mesh_square(arr, x, y, size):
    s = int(size / 2)

    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

    return np.logical_or(arr, logical_and([xx > x - s, xx < x + s, xx % 2 == 1, yy > y - s, yy < y + s, yy % 2 == 1]))

def add_triangle(arr, x, y, size):
    s = int(size / 2)

    triangle = np.tril(np.ones((size, size), dtype=bool))

    arr[x-s:x-s+triangle.shape[0],y-s:y-s+triangle.shape[1]] = triangle

    return arr

def add_circle(arr, x, y, size):
    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]
    circle = (xx - x) ** 2 + (yy - y) ** 2
    new_arr = np.logical_or(arr, np.logical_and(circle < size, circle > size * 0.5))

    return new_arr

def get_random_location(shape):
    x = np.random.randint(shape[0] * 0.2, shape[0] * 0.8)
    y = np.random.randint(shape[1] * 0.2, shape[1] * 0.8)
    size = np.random.randint(shape[0] * 0.2, shape[0] * 0.4)

    return (x, y, size)
