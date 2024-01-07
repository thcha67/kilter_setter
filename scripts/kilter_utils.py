import sqlite3
import re


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
    "12" : "green",
    "13" : "blue",
    "14" : "pink",
    "15" : "yellow",
}

def decompose_frames(frames):
    pattern = re.compile(r'p(\d{4})r(\d{2})')
    return pattern.findall(frames)

def frames_to_holds(frames):
    placements_and_colors = decompose_frames(frames)
    conn = sqlite3.connect('data/raw_database.sqlite3')
    cur = conn.cursor()
    holds = []
    for placement_and_color in placements_and_colors:
        placement_id = placement_and_color[0]
        hole_id = cur.execute(f'SELECT hole_id FROM placements WHERE id = {placement_id}').fetchone()[0]
        coord = cur.execute(f'SELECT x, y FROM holes WHERE id = {hole_id}').fetchone()
        color = color_translations[placement_and_color[1]]
        holds.append((coord, color))
    conn.close()
    return holds

def get_frames_by_name(name):
    conn = sqlite3.connect('data/raw_database.sqlite3')
    cur = conn.cursor()
    frames = cur.execute(f'SELECT frames FROM climbs WHERE name = "{name}"').fetchone()[0]
    conn.close()
    return frames

frames = get_frames_by_name("Dodge Grand Caravan")
holds = frames_to_holds(frames)

print(holds)
