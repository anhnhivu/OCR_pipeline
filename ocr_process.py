import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT
#OrderedDict: dictionary subclass that remembers the order that keys were first inserted
from collections import OrderedDict
import copy
import math
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pytesseract
import re
from jellyfish import jaro_distance
import pyproj
import psycopg2

############### SETUP VARIABLE ###############
text_threshold = 0.7
low_text = 0.4
link_threshold =0.4
# cuda = True
cuda=False
canvas_size =1280
mag_ratio =1.5
#if text image present curve --> poly=true
poly=False
refine=False
show_time=False
refine_net = None

############### CRAFT ###############
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    ##print("X: ",x)
  
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    #print("Score_text: ", score_text)
    #print("Score_link: ", score_link)

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()
    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]
    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)
    #print("Render image: ", render_img)
    # plt.imshow(ret_score_text)
    #print("Bounding Box: ", polys)

    # if show_time : 
    #     print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

############### G-DBSCAN ###############
class Point:
    '''
    Each point have 2 main values: coordinate(lat, long) and cluster_id
    '''
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id
        self.cluster_id = UNCLASSIFIED

    def __repr__(self):
        return '(x:{}, y:{}, id:{}, cluster:{})' \
            .format(self.x, self.y, self.id, self.cluster_id)

#In G-DBScan we use elip instead of circle to cluster (because we mainly use for horizontal text image --> elip is more useful)
def n_pred(p1, p2):
#     return (p1.x - p2.x)**2/160000 + (p1.y - p2.y)**2/2500 <= 1
#     print(p1.x -p2.x)
#     print(p1.y -p2.y)
#     return (p1.x - p2.x)**2/50000 + (p1.y - p2.y)**2/1500 <= 1
#     return (p1.x - p2.x)**2/20000 + (p1.y - p2.y)**2/1300 <= 1
     return (p1.x - p2.x)**2/2000 + (p1.y - p2.y)**2/130 <= 1
#     return (p1.x - p2.x)**2/3500 + (p1.y - p2.y)**2/150 <= 1
#     return (p1.x - p2.x)**2/7000 + (p1.y - p2.y)**2/1300 <= 1
#     return (p1.x - p2.x)**2/8000 + (p1.y - p2.y)**2/300 <= 1
#     return (p1.x - p2.x)**2/17000 + (p1.y - p2.y)**2/300 <= 1
#     return (p1.x - p2.x)**2/13000 + (p1.y - p2.y)**2/250 <= 1
#    return (p1.x - p2.x)**2/15000 + (p1.y - p2.y)**2/180 <= 1


def w_card(points):
    return len(points)
UNCLASSIFIED = -2
NOISE = -1

def GDBSCAN(points, n_pred, min_card, w_card):
    points = copy.deepcopy(points)
    cluster_id = 0
    for point in points:
        if point.cluster_id == UNCLASSIFIED:
            if _expand_cluster(points, point, cluster_id, n_pred, min_card,
                               w_card):
                cluster_id = cluster_id + 1
    clusters = {}
    for point in points:
        key = point.cluster_id
        if key in clusters:
            clusters[key].append(point)
        else:
            clusters[key] = [point]
    return list(clusters.values())


def _expand_cluster(points, point, cluster_id, n_pred, min_card, w_card):
    if not _in_selection(w_card, point):
        points.change_cluster_id(point, UNCLASSIFIED)
        return False

    seeds = points.neighborhood(point, n_pred)
    if not _core_point(w_card, min_card, seeds):
        points.change_cluster_id(point, NOISE)
        return False

    points.change_cluster_ids(seeds, cluster_id)
    seeds.remove(point)

    while len(seeds) > 0:
        current_point = seeds[0]
        result = points.neighborhood(current_point, n_pred)
        if w_card(result) >= min_card:
            for p in result:
                if w_card([p]) > 0 and p.cluster_id in [UNCLASSIFIED, NOISE]:
                    if p.cluster_id == UNCLASSIFIED:
                        seeds.append(p)
                    points.change_cluster_id(p, cluster_id)
        seeds.remove(current_point)
    return True


def _in_selection(w_card, point):
    return w_card([point]) > 0


def _core_point(w_card, min_card, points):
    return w_card(points) >= min_card


class Points:
    'Contain list of Point'
    def __init__(self, points):
        self.points = points

    def __iter__(self):
        for point in self.points:
            yield point

    def __repr__(self):
        return str(self.points)

    def get(self, index):
        return self.points[index]

    def neighborhood(self, point, n_pred):
        return list(filter(lambda x: n_pred(point, x), self.points))

    def change_cluster_ids(self, points, value):
        for point in points:
            self.change_cluster_id(point, value)

    def change_cluster_id(self, point, value):
        index = (self.points).index(point)
        self.points[index].cluster_id = value

    def labels(self):
        return set(map(lambda x: x.cluster_id, self.points))

def similar_value(lst1, lst2):
  val = 0
  for coor in lst1:
    for element in lst2:
      val += jaro_distance(element,coor)
  return val/len(lst1)

def eliminate(lst1, lst2):
  lst2 = lst1.copy()
  for element in lst1:
    if re.search('\.', element) is None:
      lst2.remove(element)
  return lst2

# print(coor_arr_1)
# print(coor_arr_2)
def insert_point(lst):
  for i in range(len(lst)):
    if len(lst[i]) == 9:
      lst[i] = lst[i][:7] + '.' + lst[i][7:]
    else:
      lst[i] = lst[i][:-2] + '.' + lst[i][-2:]

def findX(coor_arr, coordinates):
  length = 0
  for coor in coor_arr:
    length += len(str(coor))
  try:
    return length/len(coor_arr)
  except ZeroDivisionError:
    return 0

def str_similarity(coor_arr, coordinates):
  sim_arr = []
  temp = []
  for coor in coor_arr:
    for ele in coordinates:
      if ele != str(coor) and ele not in coor_arr:
        temp.append(jaro_distance(str(coor), ele))
    sim_arr.append(max(temp))
    temp = []
  return sim_arr

def vn2k_to_wgs84(coordinate,crs): 
    """
    Đây là hàm chuyển đổi cặp toạ độ x, y theo vn2k sang kinh độ , vĩ độ theo khung toạ độ của Google Map 
    Công thức này được cung cấp bởi thư viện pyproj 

    
    Input:
    
        - ( x, y ) : TUPLE chứa cặp toạ độ x và y theo đơn vị float 
        - crs : INT - id (mã) vùng chứa cặp toạ độ x, y theo toạ độ Google

    Output: 

        - (longitude, latitude): TUPLE chứa cặp kinh độ - vĩ độ theo toạ độ Google Map


    """
    new_coordinate = pyproj.Transformer.from_crs(
        crs_from=crs, crs_to=4326, always_xy=True).transform(coordinate[1], coordinate[0])

    return new_coordinate

    ################### WORD DETECTION ###################
    # Initialize CRAFT parameters
def processing(file_name, crs):
    #path of file pre-trained model of Craft
    trained_model_path = './craft_mlt_25k.pth'
    #trained_model_path = './vgg16.ckpt'

    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(trained_model_path, map_location='cpu')))
    net.eval()

    # Load image from its path
    image_path = f'./imgtxtenh/pre_{file_name}'
    image = imgproc.loadImage(image_path)

    fig2 = plt.figure(figsize = (10,10)) # create a 10 x 10 figure 
    ax3 = fig2.add_subplot(111)
    ax3.imshow(image, interpolation='none')
    ax3.set_title('larger figure')
    plt.show()

    poly=False
    refine=False
    show_time=False
    refine_net = None
    bboxes, polys, score_text = test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net)
    file_utils.saveResult(image_path, image[:,:,::-1], bboxes, dirname='./craft_result/')
    # Compute coordinate of central point in each bounding box returned by CRAFT
    #Purpose: easier for us to make cluster in G-DBScan step
    poly_indexes = {}
    central_poly_indexes = []
    for i in range(len(polys)):
        poly_indexes[i] =  polys[i]
        x_central = (polys[i][0][0] + polys[i][1][0] +polys[i][2][0] + polys[i][3][0])/4
        y_central = (polys[i][0][1] + polys[i][1][1] +polys[i][2][1] + polys[i][3][1])/4
        central_poly_indexes.append({i: [int(x_central), int(y_central)]})

    # for i in central_poly_indexes:
    #   print(i)

    # For each of these cordinates convert them to new Point instances
    X = []

    for idx, x in enumerate(central_poly_indexes):
        point = Point(x[idx][0],x[idx][1], idx)
        X.append(point)

    # Cluster these central points
    clustered = GDBSCAN(Points(X), n_pred, 1, w_card)

    # Create bounding box for each cluster with 4 points
    #Purpose: Merge words in 1 cluster into 1 bounding box
    cluster_values = []
    for cluster in clustered:
        sort_cluster = sorted(cluster, key = lambda elem: (elem.x, elem.y))
        max_point_id = sort_cluster[len(sort_cluster) - 1].id
        min_point_id = sort_cluster[0].id
        max_rectangle = sorted(poly_indexes[max_point_id], key = lambda elem: (elem[0], elem[1]))
        min_rectangle = sorted(poly_indexes[min_point_id], key = lambda elem: (elem[0], elem[1]))

        right_above_max_vertex = max_rectangle[len(max_rectangle) -1]
        right_below_max_vertex = max_rectangle[len(max_rectangle) -2]
        left_above_min_vertex = min_rectangle[0] 
        left_below_min_vertex = min_rectangle[1]
        
        if (int(min_rectangle[0][1]) > int(min_rectangle[1][1])): 
            left_above_min_vertex = min_rectangle[1]
            left_below_min_vertex =  min_rectangle[0]
        if (int(max_rectangle[len(max_rectangle) -1][1]) < int(max_rectangle[len(max_rectangle) -2][1])):
            right_above_max_vertex = max_rectangle[len(max_rectangle) -2]
            right_below_max_vertex = max_rectangle[len(max_rectangle) -1]
            
            
        cluster_values.append([left_above_min_vertex, left_below_min_vertex, right_above_max_vertex, right_below_max_vertex])
        
        # for p in cluster_values:
        #   print(p)

    file_utils.saveResult(image_path, image[:,:,::-1], cluster_values, dirname='./cluster_result/')
    img = np.array(image[:,:,::-1])        
    ocr_res = []
    plain_txt = ""
    for i, box in enumerate(cluster_values):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)

        rect = cv2.boundingRect(poly)
        x,y,w,h = rect
        croped = img[y:y+h, x:x+w].copy()
        
        # Preprocess croped segment
        croped = cv2.resize(croped, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)
        croped = cv2.cvtColor(croped, cv2.COLOR_BGR2GRAY)
        croped = cv2.GaussianBlur(croped, (3, 3), 0)
        croped = cv2.bilateralFilter(croped,5,25,25)
        croped = cv2.dilate(croped, None, iterations=1)
        croped = cv2.threshold(croped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #     croped = cv2.threshold(croped, 90, 255, cv2.THRESH_BINARY)[1]
        croped = cv2.cvtColor(croped, cv2.COLOR_BGR2RGB)
        custom_oem_psm_config = r'--oem 1 --psm 12'
        # print("--------")
        # print(pytesseract.image_to_string(croped, lang='eng'))
        plain_txt += "--------\n"
        plain_txt += pytesseract.image_to_string(croped, lang='eng', config=custom_oem_psm_config)

    copy_plain_txt = plain_txt
    # plain_txt = re.sub(r"b", "6", plain_txt)
    plain_txt = re.sub(r"\$", "5", plain_txt)
    plain_txt = re.sub(r"%", "7", plain_txt)
    plain_txt = re.sub(r"Y", "5", plain_txt)
    plain_txt = re.sub(r"W", "99", plain_txt)
    plain_txt = re.sub(r"£", "1", plain_txt)
    plain_txt = re.sub(r"\)", "1", plain_txt)
    plain_txt = re.sub(r"\}", "1", plain_txt)
    plain_txt = re.sub(r"\|", "1", plain_txt)

    # print(plain_txt)
    # return 0
    #Localization
    init_patterns_1 = re.compile(r'TOA\sDO', re.IGNORECASE)
    init_patterns_2 = re.compile(r'\w{0,2}\d{5,}', re.IGNORECASE)
    term_patterns = re.compile(r'\n[^\-\d]{10,}', re.IGNORECASE)
    coor_patterns = re.compile(r'\d+\s*[\d]*\s*[\d\.]*', re.IGNORECASE)
    coordinates = coor_patterns.findall(plain_txt)
    for i in range(len(coordinates)):
        coordinates[i] = re.sub('\n','',coordinates[i])
        coordinates[i] = re.sub('\x0c','',coordinates[i])
        coordinates[i] = re.sub(r'\s','',coordinates[i])
    # print(coordinates)
    # return 0
    temp_arr = coordinates.copy()
    for i in range(len(temp_arr)):
        try:
            # print(float(temp_arr[i]))
            if len(temp_arr[i]) <= 7:
                coordinates.remove(temp_arr[i])
        except ValueError:
            coordinates.remove(temp_arr[i])
    print(coordinates)

    cluster_arr = [[coor] for coor in coordinates]
    for i in range(len(coordinates)):
        for coor in coordinates:
            if cluster_arr[i][0] != coor and cluster_arr[i][0][0] == coor[0] and cluster_arr[i][0][1] == coor[1] and cluster_arr[i][0][2] == coor[2]:
                cluster_arr[i].append(coor)
    # print(cluster_arr)

    cluster_lens = []
    for cluster in cluster_arr:
        cluster_lens.append(len(cluster))
    # print(cluster_lens)

    try:
        max_len = max(cluster_lens)
    except ValueError:
        max_len = 0
    coor_arr_1 = []
    for cluster in cluster_arr:
        if max_len == len(cluster):
            coor_arr_1 = cluster
            break
    # print(coor_arr_1)

    cluster_arr = []
    for coor in coordinates:
        if coor not in coor_arr_1:
            cluster_arr.append([coor])
    # print(cluster_arr)

    for i in range(len(cluster_arr)):
        for coor in coordinates:
            if coor not in coor_arr_1 and cluster_arr[i][0] != coor and cluster_arr[i][0][0] == coor[0] and cluster_arr[i][0][1] == coor[1] and cluster_arr[i][0][2] == coor[2]:
                cluster_arr[i].append(coor)
    # print(cluster_arr)

    cluster_lens = []
    for cluster in cluster_arr:
        cluster_lens.append(len(cluster))
    # print(cluster_lens)

    try:
        max_len = max(cluster_lens)
    except ValueError:
        max_len = 0
    
    # print(cluster_arr)
    coor_arr_2 = []
    similar_cluster_arr = []
    temp = 0
    for cluster in cluster_arr:
        if max_len == len(cluster):
            temp += 1
            coor_arr_2 = cluster
            similar_cluster_arr.append(cluster)
    if temp > 1:
        similar_val_arr = []
        for cluster in similar_cluster_arr:
            similar_val_arr.append(similar_value(cluster, coor_arr_1))
        right_index = np.where(similar_val_arr == np.amin(similar_val_arr))[0][0]
        coor_arr_2 = similar_cluster_arr[right_index]
    # print(coor_arr_2)

    temp_lst = []


    if len(eliminate(coor_arr_1, temp_lst)) != 0:
        coor_arr_1 = eliminate(coor_arr_1, temp_lst)
    else:
        insert_point(coor_arr_1)
    # print('Arr 1 after remove:')
    # print(coor_arr_1)

    if len(eliminate(coor_arr_2, temp_lst)) != 0:
        coor_arr_2 = eliminate(coor_arr_2, temp_lst)
    else:
        insert_point(coor_arr_2)
    # print('Arr 2 after remove:')
    # print(coor_arr_2)

    X = []
    Y = []


    if findX(coor_arr_1, coordinates) > findX(coor_arr_2, coordinates):
        X = coor_arr_1
        Y = coor_arr_2
    else:
        X = coor_arr_2
        Y = coor_arr_1 

    print('X: ' + str(X))
    print('Y: ' + str(Y))

    temp_arr = []
    for coor in X:
        try:
            float(coor)
            temp_arr.append(float(coor))
        except ValueError:
            pass
    X = temp_arr
    temp_arr = []
    for coor in Y:
        try:
            float(coor)
            temp_arr.append(float(coor))
        except ValueError:
            pass
    Y = temp_arr

    sim_arr = str_similarity(X, coordinates)
    sim_arr = np.array(sim_arr)
    try:
        optimal_index = np.where(sim_arr == np.amax(sim_arr))[0][0]
        x = X[optimal_index]
    except ValueError:
        x = 0

    sim_arr = str_similarity(Y, coordinates)
    sim_arr = np.array(sim_arr)
    try:
        optimal_index = np.where(sim_arr == np.amax(sim_arr))[0][0]
        y = Y[optimal_index]
    except ValueError:
        y = 0

    print('Most likely to be x: ' + str(x))
    print('Most likely to be y: ' + str(y))

    #################### VN2K TO WGS83 ####################

    y,x = vn2k_to_wgs84((x,y), crs)
    print((x,y))
    return (x,y)

# processing('test_16.jpg', 9210)