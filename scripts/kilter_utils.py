import sqlite3
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
import torch

grade_translations = {
    "10" : "4a/V0",
    "11" : "4b/V0",
    "12" : "4c/V0",
    "13" : "5a/V1",
    "14" : "5b/V1",
    "15" : "5c/V2",
    "16" : "6a/V3",
    "17" : "6a+/V3",
    "18" : "6b/V4",
    "19" : "6b+/V4",
    "20" : "6c/V5",
    "21" : "6c+/V5",
    "22" : "7a/V6",
    "23" : "7a+/V7",
    "24" : "7b/V8",
    "25" : "7b+/V8",
    "26" : "7c/V9",
    "27" : "7c+/V10",
    "28" : "8a/V11",
    "29" : "8a+/V12",
    "30" : "8b/V13",
    "31" : "8b+/V14",
    "32" : "8c/V15",
    "33" : "8c+/V16",
}

angle_translations = {
    "15" : "15°",
    "20" : "20°",
    "25" : "25°",
    "30" : "30°",
    "35" : "35°",
    "40" : "40°",
    "45" : "45°",
    "50" : "50°",
    "55" : "55°",
    "60" : "60°",
    "65" : "65°",
}

color_translations = {
    "12" : "#00DD00", # green
    "13" : "#00FFFF", # blue
    "14" : "#FF00FF", # pink
    "15" : "#FFA550", # orange
}

def decompose_frames(frames):
    """
    Decompose frames like "p0001r12p0002r13p0003r14p0004r15" into a list of tuples like
    [(0001, 12), (0002, 13), (0003, 14), (0004, 15)] where the first element of each tuple
    is the placement id and the second element is the color id.
    """
    pattern = re.compile(r'p(\d{4})r(\d{2})')
    return pattern.findall(frames)

def frames_to_holes(frames):
    """
    Query the database to get a list of holds and colors from a string of frames.
    """
    placements_and_colors = decompose_frames(frames)
    conn = sqlite3.connect('data/raw_database.sqlite3')
    cur = conn.cursor()
    holds = []
    for placement_and_color in placements_and_colors:
        placement_id = placement_and_color[0]
        coord = cur.execute(f'SELECT x, y FROM holes LEFT OUTER JOIN placements ON placements.hole_id = holes.id WHERE placements.id = {placement_id}').fetchone()
        color_id = placement_and_color[1]
        holds.append((coord[0], coord[1], color_id))
    conn.close()
    return holds

def get_frames_by_name(name):
    """
    Query the database to get a string of frames from a climb name.
    """
    conn = sqlite3.connect('data/raw_database.sqlite3')
    cur = conn.cursor()
    frames = cur.execute(f'SELECT frames FROM climbs WHERE name = "{name}"').fetchone()[0]
    conn.close()
    return frames


def get_all_holes_12x12():
    """
    Query the database to get a list of all holes in the 12x12 kilter. The coordinates
    are in the form (x, y) and are in the range (-20, 4) to (140, 152).
    """
    conn = sqlite3.connect('data/raw_database.sqlite3')
    cur = conn.cursor()
    # the "12" in the SELECT statement is the color_id for green holds, which is the default color when a specific boulder is not selected
    holes = cur.execute(f'SELECT x, y, "12" FROM holes WHERE product_id = 1 AND (x > 0 AND x < 144) AND (y > 0 AND y < 156)').fetchall()
    conn.close()
    return holes


def get_matrix_from_holes(holes, color_as_number=True):
    """
    Create a matrix of 0s and numbers from a list of holes. If not a 
    0, the number represents the color of the hole.
    """
    matrix = np.zeros((157, 145))
    try:
        for hole in holes:
            x = hole[0]
            y = hole[1]
            color = int(hole[2]) if color_as_number else color_translations[hole[2]]
            matrix[y, x] = color
    except IndexError:
        print("Holes out of range")

    return matrix[::4, ::4] # shape 40x37

def plot_matrix(matrix):
    """
    Plot a matrix of 0s and numbers. If not a 0, the number represents the color of the hole.
    """
    cmap = ListedColormap(['#FFFFFF', *color_translations.values()])
    bounds = [0, 11.5, 12.5, 13.5, 14.5, 15.5]

    norm = BoundaryNorm(bounds, cmap.N)
    plt.imshow(matrix, cmap=cmap, norm=norm, origin='lower')
    plt.show()

def get_most_recent_boulders_frames(n):
    """
    Query the database to get the n most recently created boulders and their frames.
    """
    conn = sqlite3.connect('data/raw_database.sqlite3')
    cur = conn.cursor()
    boulders = cur.execute(f'SELECT frames FROM climbs AND climbs.layout_id = 1 ORDER BY created_at DESC LIMIT {n}').fetchall()
    conn.close()
    return boulders

def get_most_popular_boulders_frames(n):
    """
    Query the database to get the n most popular boulders and their frames.
    """
    conn = sqlite3.connect('data/raw_database.sqlite3')
    cur = conn.cursor()
    boulders = cur.execute(f'SELECT frames FROM climbs LEFT OUTER JOIN climb_stats ON climbs.uuid = climb_stats.climb_uuid WHERE climb_stats.display_difficulty != "None" AND climbs.layout_id = 1 ORDER BY climb_stats.ascensionist_count DESC LIMIT {n}').fetchall()
    conn.close()
    return boulders

def get_most_recent_boulders_grades(n):
    """
    Query the database to get the n most recently created boulders and their grades.
    """
    conn = sqlite3.connect('data/raw_database.sqlite3')
    cur = conn.cursor()
    difficulties = cur.execute(f'SELECT display_difficulty FROM climbs LEFT OUTER JOIN climb_stats ON climbs.uuid = climb_stats.climb_uuid WHERE climb_stats.display_difficulty != "None" AND climbs.layout_id = 1 ORDER BY climbs.created_at DESC LIMIT {n}').fetchall()
    conn.close()
    return difficulties

def get_most_popular_boulders_grades(n):
    """
    Query the database to get the n most popular boulders and their grades.
    """
    conn = sqlite3.connect('data/raw_database.sqlite3')
    cur = conn.cursor()
    difficulties = cur.execute(f'SELECT display_difficulty FROM climbs LEFT OUTER JOIN climb_stats ON climbs.uuid = climb_stats.climb_uuid WHERE climb_stats.display_difficulty != "None" AND climbs.layout_id = 1 ORDER BY climb_stats.ascensionist_count DESC LIMIT {n}').fetchall()
    conn.close()
    return difficulties

def get_useable_boulders(limit=100000):
    """
    Query the database to get a list of boulders that are useable for generating problems.
    The criteria are:
    - The boulder fits in 12x12 with kickboard layout
    - The boulder is not a route (only one frame)
    - The boulder has at least 4 ascensionists
    - The boulder has a quality average higher than 2.5
    - The boulder has a been graded
    - The frames only contain "r12", "r13", "r14", "r15" (no "r2X" or "r3X") ensuring only the 4 regular colors
    """
    conn = sqlite3.connect('data/raw_database.sqlite3')
    cur = conn.cursor()
    useable_boulders = cur.execute(
        f"""
    SELECT climb_stats.angle, 
        climb_stats.difficulty_average, 
        climbs.frames
    FROM climbs 
    LEFT OUTER JOIN climb_stats 
        ON climbs.uuid = climb_stats.climb_uuid
    WHERE climbs.edge_left > 0 
    AND climbs.edge_right < 144
    AND climbs.edge_bottom > 0 
    AND climbs.edge_top < 156
    AND climbs.frames_count = 1
    AND climbs.layout_id = 1
    AND climb_stats.quality_average != "None"
    AND climb_stats.ascensionist_count > 3
    AND climb_stats.quality_average > 2.5
    AND NOT (climbs.frames LIKE '%r2%')
    AND NOT (climbs.frames LIKE '%r3%')
    AND NOT LENGTH(climbs.frames) > 280
    LIMIT {limit}
    """
    ).fetchall()
    conn.close()
    print(f'Number of useable boulders: {len(useable_boulders)}')
    return useable_boulders

def create_training_data(max_samples=100000, dtype="float16", save=True, name_inputs=None, name_targets=None):
    """
    Creates and saves two numpy arrays. The first array  is the inputs, with shape (num_boulders, 2) 
    where the first column is the angle and the second column is the grade.
    The second array is the targets, with shape (num_boulders, 40, 37) where each row is a matrix 
    of 0s and numbers. If not a 0, the number represents the color of the hole such that
    12 = green, 13 = blue, 14 = pink, 15 = orange.
    """
    useable_boulders_with_frames = get_useable_boulders(max_samples)
    num_boulders = len(useable_boulders_with_frames)
    all_angles = []
    all_grades = []
    all_holes = []

    for i in tqdm(useable_boulders_with_frames):
        frames = i[2]
        holes = frames_to_holes(frames)
        matrix = get_matrix_from_holes(holes)
        all_angles.append(i[0])
        all_grades.append(i[1])
        all_holes.append(matrix)
    
    all_angles = np.array(all_angles, dtype=dtype).reshape((num_boulders, 1))  # shape (num_boulders, 1)
    all_grades = np.array(all_grades, dtype=dtype).reshape((num_boulders, 1))  # shape (num_boulders, 1)
    inputs = np.concatenate((all_angles, all_grades), axis=1)  # shape (num_boulders, 2)
    targets = np.array(all_holes, dtype=dtype) # shape (num_boulders, 40, 37)

    if save:
        np.save(f'data/{name_inputs or "inputs16"}.npy', inputs)
        np.save(f'data/{name_targets or "targets16"}.npy', targets)
    return inputs, targets

def load_training_data(device=None, path_input=None, path_target=None):
    """
    Load the training data from a numpy array.
    """
    path_input = path_input or 'data/inputs.npy'
    path_target = path_target or 'data/targets.npy'
    inputs = torch.tensor(np.load(path_input), dtype=torch.float32)
    targets = torch.tensor(np.load(path_target), dtype=torch.float32)
    if device:
        inputs = inputs.to(device)
        targets = targets.to(device)
    return inputs, targets

def normalize_data(inputs, targets):
    return inputs / 70, targets / 70

def denormalize_data(inputs, targets):
    return inputs * 70, targets * 70

def holes_from_matrix(matrix):
    """
    Create a list of holes from a matrix (40x37) of 0s and numbers. If not a 0, the number represents the color of the hole.
    """
    holes = []
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            color = matrix[y, x]
            if color > 11.5:
                holes.append((x, y, color))
    return holes

def map_targets(targets):
    mapping = {0: 0, 12: 1, 13: 2, 14: 3, 15: 4}
    return torch.tensor(np.vectorize(mapping.get)(targets), dtype=torch.float32)
    
def boulder_holes_to_4d_tensor(boulder):
    """
    Converts a list tuples of holes to a 4d tensor.
    """
    mapping = {12: 1, 13: 2, 14: 3, 15: 4}
    matrix = torch.zeros((4, 157, 145))
    for x, y, color in boulder:
        color = mapping[int(color)]
        matrix[color - 1, y, x] = 1
    return matrix[:, ::4, ::4]

def tensor_data_4d(max_samples=100000):
    useable_boulders_with_frames = get_useable_boulders(max_samples)
    big_tensor = []
    difficulties = []
    for i in tqdm(useable_boulders_with_frames):
        frames = i[2]
        difficulty = i[1]
        holes = frames_to_holes(frames)
        tensor = boulder_holes_to_4d_tensor(holes)
        big_tensor.append(tensor)
        difficulties.append(difficulty)
    np.save('data/diff.npy', np.array(difficulties))
    np.save('data/data.npy', torch.stack(big_tensor))
    return torch.tensor(difficulties), torch.stack(big_tensor)

def create_sequences(max_samples=100000):
    useable_boulders_with_frames = get_useable_boulders(max_samples)
    mapping = {12: 1, 13: 2, 14: 3, 15: 4}
    sequences = []
    for i in tqdm(useable_boulders_with_frames):
        sequence = []
        frames = i[2]
        holes = frames_to_holes(frames)
        for x, y, color in holes:
            color = int(color)
            color = mapping[color]
            sequence.append([x, y, color])
        sequences.append(sequence)
    import pickle
    with open('data/sequences.pkl', 'wb') as f:
        pickle.dump(sequences, f)
    return sequences

if __name__ == '__main__':
    create_sequences(100000)
    # import sys
    # np.set_printoptions(suppress=True,linewidth=np.nan,threshold=sys.maxsize)
    # frames = get_frames_by_name("Dodge Grand Caravan")
    # holes_DGC = frames_to_holes(frames)
    # matrix_DGC = get_matrix_from_holes(holes_DGC)
    # matrix_DGC = map_targets(matrix_DGC)

    # plot_matrix(matrix_DGC)

    # all_holes = get_all_holes_12x12()
    # for hole in all_holes:
    #     plt.scatter(hole[0], hole[1], c=color_translations[hole[2]])
    # plt.show()
    # matrix_all_holes = get_matrix_from_holes(all_holes)
    # plot_matrix(matrix_all_holes)

    # recent_1000_boulders = get_most_recent_boulders_frames(1000)
    # lengths = []
    # for boulder in recent_1000_boulders:
    #     holes = frames_to_holes(boulder[0])
    #     num_holes = len(holes)
    #     lengths.append(num_holes)
    # plt.hist(lengths, bins=50)
    # plt.show()

    # recent_1000_boulders = get_most_popular_boulders_grades(1000)
    # difficulties = [round(float(i[0])) for i in recent_1000_boulders]
    # plt.hist(difficulties, bins=21)
    # ticks = range(min(difficulties), max(difficulties))
    # plt.xticks(ticks, [grade_translations[str(i)] for i in ticks])
    # plt.show()

    # recent_1000_frames = get_most_popular_boulders_frames(30000)
    # matrices = [get_matrix_from_holes(frames_to_holes(i[0])) for i in recent_1000_frames]
    # matrix = np.sum(matrices, axis=0)
    # plt.imshow(matrix, cmap='inferno', origin='lower')
    # plt.show()

    #create_training_data(name_inputs="inputs", name_targets="targets")

    # inputs, targets = load_training_data()

    # num_holds_per_target = np.where(targets > 0, 1, 0).sum(axis=(1, 2))
    # #print(f'Number of holds: {num_holds_per_target}')
    # plt.plot(inputs[:, 1], num_holds_per_target, 'o')
    # plt.ylim(0, 40)
    # plt.show()
    # _, data = tensor_data_4d(1000)
    # data = data.mean(axis=0)
    # plt.subplot(2, 2, 1)
    # plt.imshow(data[0], cmap='inferno', origin='lower')
    # plt.subplot(2, 2, 2)
    # plt.imshow(data[1], cmap='inferno', origin='lower')
    # plt.subplot(2, 2, 3)
    # plt.imshow(data[2], cmap='inferno', origin='lower')
    # plt.subplot(2, 2, 4)
    # plt.imshow(data[3], cmap='inferno', origin='lower')

    # plt.show()
    # diff = torch.tensor(np.load("data/diff.npy"), dtype=torch.float32)
    # data = torch.tensor(np.load("data/data.npy"), dtype=torch.float32)
    # print(diff.shape)
    # print(data.shape)
    pass
