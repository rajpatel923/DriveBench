"""LiDAR/radar -> text builders for the Stage 1 "words" arm.

Each builder writes data/sensor_text/<source>.json ({frame_token: text}) and
<source>.objects.json (the structured objects behind the text), all through
the one shared formatter in format.py.
"""
