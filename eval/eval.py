import os
import sys
import glob
import cv2
from tqdm import tqdm
from utils import compactCode, Search

search_model = compactCode(centers_path="../centers/centers.h5py",
                            pq_centers_path="../centers/pq_centers.h5py",
                            codes_path="../centers/codes",
                            codes_name="../centers/codes_name")
model = Search(search_model)


def compute_ap(pos, junk, rank_list):
    old_recall = 0.0
    old_precision = 1.0
    ap = 0.0
    aps = {
        100: 0.0,
        200: 0.0,
        400: 0.0,
        800: 0.0
    }
    intersect_size = 0
    j = 0
    for i, rank in enumerate(rank_list):
        if rank in junk:
            continue
        if rank in pos:
            intersect_size += 1
        recall = intersect_size / len(pos)
        precision = intersect_size / (j + 1.0)
        ap += (recall - old_recall) * ((old_precision + precision) / 2.0)
        old_recall = recall
        old_precision = precision
        j += 1
        if i in [100, 200, 400, 800]:
            aps[i] = ap
        if i == 800:
            return aps

    return aps

if __name__ == "__main__":
    gt_files = '../groundtruth/'
    query_list = glob.glob('../groundtruth/*query*')

    mAPs = {
        100: 0.0,
        200: 0.0,
        400: 0.0,
        800: 0.0,
    }

    for query in tqdm(query_list):

        # search rank list
        with open(query, 'r') as f:
            content = f.readlines()
            contents = content[0][:-1].split(' ')
            names =  contents[0]
            names = names[5:]
            coords = [int(float(t)) for t in contents[1:]]

            image_dir = '../dataset/{}.jpg'.format(names)
            model.search(image_dir, coords)
            rank_list = model.rank_list

        # get pos and neg arr
        pos = []
        junk = []
        query_name = query.split('/')[-1].split('_')
        query_name = '_'.join(query_name[:-1])
        pos_names = glob.glob("../groundtruth/*{}*o*".format(query_name))
        for pos_name in pos_names:
            with open(pos_name, 'r') as f:
                content = f.readlines()
                content = [c[:-1] for c in content]
                pos.extend(content)

        junk_names = glob.glob("../groundtruth/*{}*junk*".format(query_name))
        for junk_name in junk_names:
            with open(junk_name, 'r') as f:
                content = f.readlines()
                content = [c[:-1] for c in content]
                junk.extend(content)

        aps = compute_ap(pos, junk, rank_list)
        for key in aps:
            mAPs[key] += aps[key]


    for key in mAPs:
        mAPs[key] /= 55
    print('mAPs at 100, 200, 400, 800\n', mAPs)



