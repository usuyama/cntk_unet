import numpy as np

np.random.seed(1234)

def generate_random_data(shape, count):
    x, y = zip(*[generate_img_and_mask((shape[1], shape[2])) for i in range(0, count)])

    X = np.reshape(np.array(x), (count, 1, shape[1], shape[2]))
    Y = np.reshape(np.array(y), (count, 1, shape[1], shape[2]))

    return X.astype(np.float32), Y.astype(np.float32)

def generate_img_and_mask(shape):
    triangle_location = get_random_location(shape)
    circle_location1 = get_random_location(shape, zoom=0.7)
    circle_location2 = get_random_location(shape, zoom=0.7)
    mesh_location = get_random_location(shape)
    square_location = get_random_location(shape)
    plus_location = get_random_location(shape)

    # Create input image
    arr = np.zeros(shape, dtype=bool)
    arr = add_triangle(arr, *triangle_location)
    arr = add_circle(arr, *circle_location1)
    arr = add_circle(arr, *circle_location2, fill=True)
    arr = add_mesh_square(arr, *mesh_location)
    arr = add_filled_square(arr, *square_location)
    arr = add_plus(arr, *plus_location)

    # Create target mask
    mask = np.zeros(shape, dtype=bool)
    mask = add_circle(mask, *circle_location2, fill=True)

    return arr, mask

def add_square(arr, x, y, size):
    s = int(size / 2)
    arr[x-s,y-s:y+s] = True
    arr[x+s,y-s:y+s] = True
    arr[x-s:x+s,y-s] = True
    arr[x-s:x+s,y+s] = True

    return arr

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

def add_circle(arr, x, y, size, fill=False):
    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]
    circle = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    new_arr = np.logical_or(arr, np.logical_and(circle < size, circle >= size * 0.7 if not fill else True))

    return new_arr

def add_plus(arr, x, y, size):
    s = int(size / 2)
    arr[x-1:x+1,y-s:y+s] = True
    arr[x-s:x+s,y-1:y+1] = True

    return arr

def get_random_location(shape, zoom=1.0):
    x = np.random.randint(shape[0] * 0.2, shape[0] * 0.8)
    y = np.random.randint(shape[1] * 0.2, shape[1] * 0.8)
    size = int(np.random.randint(shape[0] * 0.06, shape[0] * 0.12) * zoom)

    return (x, y, size)
