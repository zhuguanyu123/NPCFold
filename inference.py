# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sys import path
import csv
import codecs
from torch.utils.data import DataLoader
from data_loader import test_data_generator
import numpy as np
import torch
from test_attention import result2
# import calculate_similarity
# from train_randforest import randf
import test3

def retrieve2(model, queries, cross_entropy_flag, img_size, infer_batch_size):

    query_img_dataset = test_data_generator(queries, img_size=img_size)
    query_loader = DataLoader(query_img_dataset, batch_size=infer_batch_size, shuffle=False, num_workers=4,
                              pin_memory=True)
    model.eval()
    model.cuda()
    query_paths, query_vecs = batch_process2(model, cross_entropy_flag, query_loader)
    file_handle=open("./LE_pairs.txt",'a') 
    for index, vec in enumerate(query_vecs):
        query_name = query_paths[index]
        query_vec = vec.reshape(1024)
        for index2, vec2 in enumerate(query_vecs):
            if index != index2:
                tempt_name = query_paths[index2]
                tempt_vec = vec2.reshape(1024)
                score = np.matmul(query_vec,tempt_vec)
                # score = 1 - distance.pdd(query_vec,tempt_vec)
                # score = calculate_similarity.calculate_pierxun(query_vec,tempt_vec)
                file_handle.write(query_name+" "+tempt_name+" "+str(score)+"\n")
    file_handle.close()
    # randf()
    return result2()


def retrieveimage(model, queries, cross_entropy_flag, img_size, infer_batch_size):

    query_img_dataset = test_data_generator(queries, img_size=img_size)
    query_loader = DataLoader(query_img_dataset, batch_size=infer_batch_size, shuffle=False, num_workers=4,
                              pin_memory=True)
    model.eval()
    model.cuda()
    query_paths, query_vecs = batch_process2(model, cross_entropy_flag, query_loader)
    dict = {}
    for i, path in enumerate(query_paths):
        dict[path] = query_vecs[i]
        # print(query_vecs[i])
    import pandas as pd
    # data = np.load('./triplet_bi/newnames.pkl',allow_pickle=True).values
    with open("./triplet_bi/newnames.txt", "r") as f:
        paths = f.readlines()
    newname = []
    label = []
    for path in paths:
        name = path.split(" ")[0]
        newname.append(dict[name])
        label.append(path.split(" ")[1].strip('\n'))
    data_tsne = test3.prepare_tsne(2,newname,label)
    test3.plot_animation_2d(data_tsne, write=True,name='newtsne2.jpg')
    # file_csv = codecs.open('./data/result.csv', 'w+', 'utf-8')  # 追加
    # writer = csv.writer(file_csv, delimiter='', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    # df = pd.DataFrame(newdict)
    # df.to_csv('./data/result.csv')

    # df = pd.DataFrame(newdict)
    # df.to_pickle('foo2.pkl')

    # randf()
    # return result()

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
 
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def retrieve4(model, queries, cross_entropy_flag, img_size, infer_batch_size):

    query_img_dataset = test_data_generator(queries, img_size=img_size)
    query_loader = DataLoader(query_img_dataset, batch_size=infer_batch_size, shuffle=False, num_workers=4,
                              pin_memory=True)
    model.eval()
    model.cuda()
    feature_vecs = []
    img_paths = []
    vec_dict = {}
    for data in query_loader:
        paths, inputs = data
        if cross_entropy_flag:
            feature_vec,_ = _get_feature(model, inputs.cuda())
        else:
            feature_vec = _get_feature(model, inputs.cuda())
        feature_vec = feature_vec.detach().cpu().numpy()  # (batch_size, channels)
        for i in range(feature_vec.shape[0]):
            feature_vecs.append(feature_vec[i])
        for path in paths:
            img_paths.append(path.split('/')[-1].split('.')[0])
    print("1",len(img_paths),len(feature_vecs))
    file_handle=open("./LE_pairs.txt",'a') 
    for index, vec in enumerate(feature_vecs):
        query_name = img_paths[index]
        query_vec = vec.reshape(1024)
        for index2, vec2 in enumerate(feature_vecs):
            if index != index2:
                tempt_name = img_paths[index2]
                tempt_vec = vec2.reshape(1024)
                score = np.matmul(query_vec,tempt_vec)
                if (query_name,tempt_name) in vec_dict.keys():
                    if vec_dict[(query_name,tempt_name)]<score:
                        vec_dict[(query_name,tempt_name)] = score
                else:
                    vec_dict[(query_name,tempt_name)] = score
    print("2",len(vec_dict))
    for key in vec_dict.keys():
        file_handle.write(key[0]+" "+key[1]+" "+str(vec_dict[key])+"\n")
        # print("3",key[0],key[1],vec_dict[key])
    file_handle.close()
    return result2()
    

def retrieve(model, queries, db, img_size, infer_batch_size):

    query_paths = queries
    reference_paths = db

    query_img_dataset = test_data_generator(queries, img_size=img_size)
    reference_img_dataset = test_data_generator(db, img_size=img_size)

    query_loader = DataLoader(query_img_dataset, batch_size=infer_batch_size, shuffle=False, num_workers=4,
                              pin_memory=True)
    reference_loader = DataLoader(reference_img_dataset, batch_size=infer_batch_size, shuffle=False, num_workers=4,
                                  pin_memory=True)

    model.eval()
    model.cuda()

    query_paths, query_vecs = batch_process(model, query_loader)
    reference_paths, reference_vecs = batch_process(model, reference_loader)

    assert query_paths == queries and reference_paths == db, "order of paths should be same"

    # DBA and AQE
    query_vecs, reference_vecs = db_augmentation(query_vecs, reference_vecs, top_k=10)
    query_vecs, reference_vecs = average_query_expansion(query_vecs, reference_vecs, top_k=5)

    sim_matrix = calculate_sim_matrix(query_vecs, reference_vecs)

    indices = np.argsort(sim_matrix, axis=1)
    indices = np.flip(indices, axis=1)

    retrieval_results = {}

    # Evaluation: mean average precision (mAP)
    # You can change this part to fit your evaluation skim
    for (i, query) in enumerate(query_paths):
        query = query.split('/')[-1].split('.')[0]
        ranked_list = [reference_paths[k].split('/')[-1].split('.')[0] for k in indices[i]]
        ranked_list = ranked_list[:1000]

        retrieval_results[query] = ranked_list

    return retrieval_results


def db_augmentation(query_vecs, reference_vecs, top_k=10):
    """
    Database-side feature augmentation (DBA)
    Albert Gordo, et al. "End-to-end Learning of Deep Visual Representations for Image Retrieval,"
    International Journal of Computer Vision. 2017.
    https://link.springer.com/article/10.1007/s11263-017-1016-8
    """
    weights = np.logspace(0, -2., top_k+1)

    # Query augmentation
    sim_mat = calculate_sim_matrix(query_vecs, reference_vecs)
    indices = np.argsort(-sim_mat, axis=1)

    top_k_ref = reference_vecs[indices[:, :top_k], :]
    query_vecs = np.tensordot(weights, np.concatenate([np.expand_dims(query_vecs, 1), top_k_ref], axis=1), axes=(0, 1))

    # Reference augmentation
    sim_mat = calculate_sim_matrix(reference_vecs, reference_vecs)
    indices = np.argsort(-sim_mat, axis=1)

    top_k_ref = reference_vecs[indices[:, :top_k+1], :]
    reference_vecs = np.tensordot(weights, top_k_ref, axes=(0, 1))

    return query_vecs, reference_vecs


def average_query_expansion(query_vecs, reference_vecs, top_k=5):
    """
    Average Query Expansion (AQE)
    Ondrej Chum, et al. "Total Recall: Automatic Query Expansion with a Generative Feature Model for Object Retrieval,"
    International Conference of Computer Vision. 2007.
    https://www.robots.ox.ac.uk/~vgg/publications/papers/chum07b.pdf
    """
    # Query augmentation
    sim_mat = calculate_sim_matrix(query_vecs, reference_vecs)
    indices = np.argsort(-sim_mat, axis=1)

    top_k_ref_mean = np.mean(reference_vecs[indices[:, :top_k], :], axis=1)
    query_vecs = np.concatenate([query_vecs, top_k_ref_mean], axis=1)

    # Reference augmentation
    sim_mat = calculate_sim_matrix(reference_vecs, reference_vecs)
    indices = np.argsort(-sim_mat, axis=1)

    top_k_ref_mean = np.mean(reference_vecs[indices[:, 1:top_k+1], :], axis=1)
    reference_vecs = np.concatenate([reference_vecs, top_k_ref_mean], axis=1)

    return query_vecs, reference_vecs


def calculate_sim_matrix(query_vecs, reference_vecs):
    query_vecs, reference_vecs = postprocess(query_vecs, reference_vecs)
    return np.dot(query_vecs, reference_vecs.T)


def batch_process(model, loader):
    feature_vecs = []
    img_paths = []
    for index, data in enumerate(loader):
        paths, inputs = data
        feature_vec = _get_feature(model, inputs.cuda())
        feature_vec = feature_vec.detach().cpu().numpy()  # (batch_size, channels)
        for i in range(feature_vec.shape[0]):
            feature_vecs.append(feature_vec[i])
        img_paths = img_paths + list(paths)

    return img_paths, np.asarray(feature_vecs)

def batch_process2(model, cross_entropy_flag, loader):

    vec_dict = {}
    for data in loader:
        # print(data)
        paths, inputs = data
        # print(paths)
        # print(inputs)
        if cross_entropy_flag:
            # _,feature_vec = _get_feature(model, inputs.cuda())
            feature_vec,_ = _get_feature(model, inputs.cuda())
        else:
            feature_vec = _get_feature(model, inputs.cuda())
        # print(feature_vec)
        feature_vec = feature_vec.detach().cpu().numpy()  # (batch_size, channels)
        # for i in range(feature_vec.shape[0]):
        #     feature_vecs.append(feature_vec[i])
        for id, path in enumerate(paths):
            path = path.split('/')[-1].split('.')[0]
            feature = np.asarray(feature_vec[id])
            if path in vec_dict.keys():
                vec_dict[path]['feature'] = (feature + vec_dict[path]['feature'])
                vec_dict[path]['num'] = vec_dict[path]['num'] + 1.0
            else:
                vec_dict[path] = {'feature':feature,'num':1.0}
            # print(vec_dict[path]['feature'].shape)
    result = []
    for key in vec_dict.keys():
        result.append(vec_dict[key]['feature'] / vec_dict[key]['num'])
    print('result:', len(result))
    return list(vec_dict.keys()), np.asarray(result)

def batch_process_image(model, test_files, cross_entropy_flag=True):

    query_img_dataset = test_data_generator(test_files, img_size=256)
    query_loader = DataLoader(query_img_dataset, batch_size=64, shuffle=False, num_workers=4,
                              pin_memory=True)
    model.eval()
    model.cuda()
    feature_vecs = []
    pathlist = []
    for data in query_loader:
        paths, inputs = data
        if cross_entropy_flag:
            feature_vec, _ = _get_feature(model, inputs.cuda())
        else:
            feature_vec = _get_feature(model, inputs.cuda())
        feature_vec = feature_vec.detach().cpu().numpy()
        for i in range(feature_vec.shape[0]):
            feature_vecs.append(feature_vec[i])
            pathlist.append(paths[i].split('/')[-2])

    return pathlist, np.asarray(feature_vecs)

def batch_process3(model, cross_entropy_flag, loader):

    feature_label = []
    vec_dict = {}
    for data in loader:
        # print(data)
        paths, inputs = data
        if cross_entropy_flag:
            # feature_vec,_ = _get_feature(model, inputs.cuda())
            _,feature_vec = _get_feature(model, inputs.cuda())
        else:
            feature_vec = _get_feature(model, inputs.cuda())
        # print(feature_vec)
        feature_vec = feature_vec.detach().cpu().numpy()  # (batch_size, channels)
        # for i in range(feature_vec.shape[0]):
        #     feature_vecs.append(feature_vec[i])
        for id, feature_path in enumerate(paths):
            # print(path)
            path = feature_path.split('/')[-1].split('.')[1].split('_')[1]
            feature = np.asarray(feature_vec[id])
            if path in vec_dict.keys():
                vec_dict[path]['feature'] = (feature + vec_dict[path]['feature'])
                vec_dict[path]['num'] = vec_dict[path]['num'] + 1.0
            else:
                vec_dict[path] = {'feature':feature,'num':1.0}
                feature_label.append(feature_path.split('/')[-1])
            # print(vec_dict[path]['feature'].shape)
    result = []
    for key in vec_dict.keys():
        result.append(vec_dict[key]['feature'] / vec_dict[key]['num'])
    # print('result:', len(result))
    return feature_label, np.asarray(result)

def _get_features_from(model, x, feature_names):
    features = {}

    def save_feature(name):
        def hook(m, i, o):
            features[name] = o.data

        return hook

    for name, module in model.named_modules():
        _name = name.split('.')[-1]
        if _name in feature_names:
            module.register_forward_hook(save_feature(_name))

    model(x)

    return features


def _get_feature(model, x):
    model_name = model.__class__.__name__

    if model_name == 'EmbeddingNetwork':
        feature = model(x)
    elif model_name == 'ResNet':
        features = _get_features_from(model, x, ['fc'])
        feature = features['fc']
    elif model_name == 'DenseNet':
        features = _get_features_from(model, x, ['classifier'])
        feature = features['classifier']
    else:
        raise ValueError("Invalid model name: {}".format(model_name))

    return feature


def postprocess(query_vecs, reference_vecs):
    """
    Postprocessing:
    1) Moving the origin of the feature space to the center of the feature vectors
    2) L2-normalization
    """
    # centerize
    query_vecs, reference_vecs = _centerize(query_vecs, reference_vecs)

    # l2 normalization
    query_vecs = _l2_normalize(query_vecs)
    reference_vecs = _l2_normalize(reference_vecs)

    return query_vecs, reference_vecs


def _centerize(v1, v2):
    concat = np.concatenate([v1, v2], axis=0)
    center = np.mean(concat, axis=0)
    return v1-center, v2-center


def _l2_normalize(v):
    norm = np.expand_dims(np.linalg.norm(v, axis=1), axis=1)
    if np.any(norm == 0):
        return v
    return v / norm
