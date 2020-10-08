"""This file is part of Splitter which is released under MIT License.

tiered_videoio.py uses opencv (cv2) to read and write files to disk. It contains
primitives to encode and decode archived and regular video formats for a tiered
storage system.
"""
import sqlite3
import random

import os
from os import path
import time
import shutil
import logging
import json
import itertools
#import threading
#import queue
from multiprocessing import Pool
#import glob
from feedback.utils.utils import Serializer


def update_headers_batch(conn, crops, name, start_frame, end_frame, start_time, end_time, ids):
    for i, id in enumerate(ids):
        clip_info = query_clip(conn, id, name)[0] # Need to update this
        updates = {}
        updates['start_frame'] = min(start_frame, clip_info[2])
        updates['end_frame'] = max(end_frame, clip_info[3])
        
        if start_time != None and clip_info[4] != 'NULL':
            updates['start_time'] = min(start_time, int(clip_info[4]))
        elif start_time != None:
            updates['start_time'] = start_time
        
        if end_time != None and clip_info[5] != 'NULL':
            updates['end_time'] = min(end_time, int(clip_info[5]))
        elif end_time != None:
            updates['end_time'] = end_time

        if i != 0:
            origin_x = crops[i - 1].x0
            origin_y = crops[i - 1].y0
            translation = clip_info[14]
            if translation == 'NULL':
                if origin_x != clip_info[6] or origin_y != clip_info[7]:
                    updates['translation'] = json.dumps([(start_time, origin_x, origin_y)])
            else:
                translation = json.loads(clip_info[14])
                if type(translation) is list:
                    if translation[-1][1] != origin_x or translation[-1][2] != origin_y:
                        translation.append((start_time, origin_x, origin_y))
                        updates['translation'] = json.dumps(translation)
                else:
                    raise ValueError('Translation object is wrongly formatted')
        
        update_clip_header(conn, id, name, updates)


def new_headers_batch(conn, all_crops, name, start_frame, end_frame, start_time, end_time, video_refs, 
                            full_dim, scaled_dim, ids = None):
    """
    Create new headers all headers for one batch. In terms of updates, we assume certain
    constraints on the system, and only update possible changes.
    """
    crops = all_crops[0]
    if ids == None:
        ids = [random.getrandbits(63) for i in range(len(crops) + 1)]        
    
    for i in range(1, len(crops) + 1):
        insert_background_header(conn, ids[0], ids[i], name)
    for i in range(0, len(crops) + 1):
        if i == 0:
            insert_clip_header(conn, ids[0], name, start_frame, end_frame, start_time, end_time, 0, 0,
                                full_dim[0], full_dim[1], scaled_dim[0], scaled_dim[1], video_refs[i], is_background=len(crops))
        else:
            origin_x = crops[i - 1].x0
            origin_y = crops[i - 1].y0
            width = crops[i - 1].x1 - crops[i - 1].x0
            height = crops[i - 1].y1 - crops[i - 1].y0
            insert_clip_header(conn, ids[i], name,start_frame, end_frame, start_time, end_time, origin_x,
                                origin_y,  width, height, width, height, video_refs[i])

    for i in range(1, len(all_crops)):
        update_headers_batch(conn, all_crops[i], name, start_time, end_time, start_frame, end_frame, ids) # TODO
    return ids

def delete_video_if_exists(conn, video_name):
    c = conn.cursor()
    c.execute("SELECT clip_id FROM clips WHERE video_name = '%s'" % video_name)
    clips = c.fetchall()
    if len(clips) == 0:
        # not exist in header file, nothing to do
        return

    clips = set().union(*map(set, clips))
    for clip in clips:
        c.execute("SELECT video_ref FROM clip WHERE clip_id = '%d' AND video_name = '%s'" % (clip, video_name))
        video_ref = c.fetchone()[0]
        try:
            os.remove(video_ref)
        except FileNotFoundError:
            logging.warning("File %s not found" % video_ref)
            
    c.execute("DELETE FROM clip WHERE video_name = '%s'" % (video_name))
    c.execute("DELETE FROM label WHERE video_name = '%s'" % (video_name))
    c.execute("DELETE FROM background WHERE video_name = '%s'" % video_name)
    c.execute("DELETE FROM lineage WHERE video_name = '%s'" % (video_name))
    conn.commit()


def move_one_file(conn, clip_id, video_name, dest_ref):
    c = conn.cursor()
    c.execute("SELECT video_ref FROM clip WHERE clip_id = '%d' AND video_name = '%s' " % (clip_id, video_name))
    video_ref = c.fetchone()[0]
    shutil.move(video_ref, dest_ref)
    c.execute("UPDATE clip SET video_ref = '%s' WHERE clip_id = '%d' AND video_name = '%s'" % (dest_ref, clip_id, video_name))
    conn.commit()


def insert_clip_header(conn, clip_id, video_name, start_frame, end_frame, start_time, end_time, origin_x, origin_y, fwidth, fheight, width, height, video_ref='', is_background = False, translation = 'NULL'):
    c = conn.cursor()
    #print((clip_id, video_name, start_time, end_time, origin_x, origin_y, fwidth, fheight, width, height, video_ref, is_background, translation, other))
    c.execute("INSERT INTO clip VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
               (clip_id, video_name, start_frame, end_frame, start_time, end_time, origin_x, origin_y, fwidth, fheight, width, height, video_ref, is_background, translation))
    conn.commit()


def insert_background_header(conn, background_id, clip_id, video_name):
    c = conn.cursor()
    c.execute("INSERT INTO background VALUES (?, ?, ?)", (background_id, clip_id, video_name))
    conn.commit()

def insert_label_header(conn, label, clip_id, video_name, data_type = 'ListStream', value = None, bbox = None, frame = None, db_name = 'label'):
    c = conn.cursor()
    if frame == None:
        frame = 'NULL'
    if value == None:
        value = 'NULL'
    if bbox == None:
        bbox = 'NULL'
    s = "INSERT INTO {} VALUES (?, ?, ?, ?, ?, ?, ?)".format(db_name)
    c.execute(s, (label, value, bbox, clip_id, video_name, data_type, frame))
    conn.commit()

def insert_lineage_header(conn, video_name, lineage, parent):
    c = conn.cursor()
    c.execute("INSERT INTO lineage VALUES (?, ?, ?)", (video_name, lineage, parent))
    conn.commit()


def delete_label_header(conn, video_name, label = None, clip_id = None): 
    c = conn.cursor()
    if label and clip_id:
        c.execute("DELETE FROM label WHERE label = '%s' AND clip_id = '%d' AND video_name = '%s' " % (label, clip_id, video_name))
    elif label:
        c.execute("DELETE FROM label WHERE label = '%s' AND video_name = '%s' " % (label, video_name))
    elif clip_id:
        c.execute("DELETE FROM label WHERE clip_id = '%d' AND video_name = '%s' " % (label, clip_id, video_name))
    else:
        raise ValueError("Neither label nor clip_id is given")

def delete_clip(conn, clip_id, video_name):
    c = conn.cursor()
    c.execute("SELECT video_ref FROM clip WHERE clip_id = '%d' AND video_name = '%s'" % (clip_id, video_name))
    video_ref = c.fetchone()[0]
    try:
        os.remove(video_ref)
    except FileNotFoundError:
        logging.warning("File %s not found" % video_ref)
    c.execute("DELETE FROM clip WHERE clip_id = '%d' and video_name = '%s' " % (clip_id, video_name))
    c.execute("DELETE FROM label WHERE clip_id = '%d' and video_name = '%s' " % (clip_id, video_name))
    c.execute("DELETE FROM background WHERE clip_id = '%d' and video_name = '%s' " % (clip_id, video_name))
    c.execute("DELETE FROM background WHERE background_id = '%d' and video_name = '%s' " % (clip_id, video_name))
    conn.commit()

def delete_video(conn, video_name):
    c = conn.cursor()
    c.execute("SELECT video_ref FROM clip WHERE video_name = '%s'" % (video_name))
    video_refs = c.fetchall()
    for ref in video_refs:
        try:
            os.remove(ref[0])
        except FileNotFoundError:
            logging.warning("File %s not found" % ref)
    c.execute("DELETE FROM clip WHERE video_name = '%s' " % video_name)
    c.execute("DELETE FROM label WHERE video_name = '%s' " % video_name)
    c.execute("DELETE FROM background WHERE video_name = '%s' " % video_name)
    c.execute("DELETE FROM lineage WHERE video_name = '%s'" % (video_name))
    conn.commit()

def delete_background(conn, background_id, video_name):
    c = conn.cursor()
    c.execute("SELECT clip_id FROM background WHERE background_id = '%d' AND video_name = '%s'" % (background_id, video_name))
    clips = c.fetchall()
    if len(clips) == 0:
        # not exist in header file, nothing to do
        return
    clips = set().union(*map(set, clips))
    clips = clips.add(background_id)
    
    for clip in clips:
        c.execute("SELECT video_ref FROM clip WHERE clip_id = '%d' AND video_name = '%s'" % (clip, video_name))
        video_ref = c.fetchone()[0]
        try:
            os.remove(video_ref)
        except FileNotFoundError:
            logging.warning("File %s not found" % video_ref)
        c.execute("DELETE FROM clip WHERE clip_id = '%d' AND video_name = '%s'" % (clip, video_name))
    
    c.execute("DELETE FROM label WHERE clip_id = '%d' video_name = '%s' " % (background_id, video_name))
    c.execute("DELETE FROM background WHERE background_id = '%d'" % background_id)
    conn.commit()


def update_clip_header(conn, clip_id, video_name, args={}):
    c = conn.cursor()
    for key, value in args.items():
        c.execute("UPDATE clip SET '%s' = '%s' WHERE clip_id = '%d' AND video_name = '%s'" % (key, value, clip_id, video_name))
    conn.commit()

def update_label_header(conn, clip_id, video_name, label, data_type, args={}):
    c = conn.cursor()
    for key, value in args.items():
        c.execute("UPDATE label SET '%s' = '%s' WHERE clip_id = '%d' AND video_name = '%s' AND label = '%s' AND type = '%s"  % (key, value, clip_id, video_name, label, data_type))
    conn.commit()

def query_clip(conn, clip_id, video_name):
    c = conn.cursor()
    c.execute("SELECT * FROM clip WHERE clip_id = '%d' AND video_name = '%s'" % (clip_id, video_name))
    result = c.fetchall()
    return result


# update queries later
def query_background(conn, video_name, background_id=None, clip_id=None):
    c = conn.cursor()
    if background_id == None and clip_id == None:
        raise ValueError("Neither background_id nor clip_id is given")
    elif background_id != None and clip_id != None:
        c.execute("SELECT * FROM background WHERE background_id = '%d' AND clip_id = '%d' AND video_name = '%s'" % (background_id, clip_id, video_name))
    elif background_id != None and clip_id == None:
        c.execute("SELECT * FROM background WHERE background_id = '%d' and video_name = '%s'" % (background_id, video_name))
    elif background_id == None and clip_id != None:
        c.execute("SELECT * FROM background WHERE clip_id = '%d' and video_name = '%s'" % (clip_id, video_name))
    result = c.fetchall()
    return result

def query_label(conn, label, video_name):
    c = conn.cursor()
    c.execute("SELECT * FROM label WHERE label = '%s' AND video_name = '%s'" % (label, video_name))
    result = c.fetchall()
    return result

def query_label_clip(conn, video_name, clip_id, label = None):
    c = conn.cursor()
    if label == None:
        c.execute("SELECT * FROM label WHERE video_name = '%s' AND clip_id = '%d" % (video_name, clip_id))
    else:
        c.execute("SELECT * FROM label WHERE label = '%s' AND video_name = '%s' AND clip_id = '%d" % (label, video_name, clip_id))
    result = c.fetchall()
    return result

def query_everything(conn, video_name):
    c = conn.cursor()
    c.execute("SELECT * FROM label WHERE video_name = '%s'" % video_name)
    result = c.fetchall()
    return result
