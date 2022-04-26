import os

import torch
import numpy as np
from inference import retrieve2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_loader import test_data_generator
from inference import batch_process3
import scipy.spatial.distance as distance
import numpy as np
from networks import EmbeddingNetwork

# plt.ion()

def save(model, ckpt_num, dir_name):
    os.makedirs(dir_name, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(dir_name, 'model_%s' % ckpt_num))
    # if torch.cuda.device_count() > 1:
    #     torch.save(model.module.state_dict(), os.path.join(dir_name, 'model_%s' % ckpt_num))
    # else:
    #     torch.save(model.state_dict(), os.path.join(dir_name, 'model_%s' % ckpt_num))
    print('model saved!')


def fit(train_loader, model, loss_fn, optimizer, scheduler, nb_epoch, cross_entropy_flag,
        device, log_interval, start_epoch=0, save_model_to='/tmp/save_model_to',train_dataset=''):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    # Save pre-trained model
    save(model, 0, save_model_to)

    test_dataset_path = './test_LEsample/'
    # test_dataset_path = './test_fold_dataset/'
    # test_dataset_path = './test/'
    test_files = [os.path.join(test_dataset_path, path) for path in os.listdir(test_dataset_path)]
    for epoch in range(0, start_epoch):
        scheduler.step()
    epochList = []
    Loss_list = []
    Sensitivity_list = []
    # print('class_to_idx:',train_dataset.class_to_idx)

    for epoch in range(start_epoch, nb_epoch):

        # Train stage
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval)
        scheduler.step()
        log_dict = {'epoch': epoch + 1,
                    'epoch_total': nb_epoch,
                    'loss': float(train_loss),
                    }
        with open("./result.txt","a") as f:
            f.writelines('epoch:' + str(epoch + 1)+'  loss:'+ str(train_loss)+ '\n')
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, nb_epoch, train_loss)
        print(message)
        print(log_dict)
        save(model, epoch + 1, save_model_to)
        retrieve2(model, test_files, cross_entropy_flag, 256, 32)
        # print('class_to_idx:',train_dataset.class_to_idx)
        # retrieve3(model, test_files, cross_entropy_flag, train_dataset, 256, 32)
        if (epoch > 2):
            epochList.append(epoch+1)
            Loss_list.append(train_loss)
            # Sensitivity_list.append(Sensitivity)
            # plt.subplot(2, 1, 1)
            plt.plot(epochList, Loss_list, '.-')
            plt.ylabel('Test loss')
            # plt.subplot(2, 1, 2)
            # plt.plot(epochList, Sensitivity_list, '.-')
            plt.xlabel('epoches')
            # plt.ylabel('Test fold accuracy')
            plt.show()
            plt.savefig("./accuracy_loss.jpg")

    


def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval):
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # for para in model.parameters():
        #     print(para)
        target = target if len(target) > 0 else None
        # print(target)
        # print(batch_idx)
        if not type(data) in (tuple, list):
            data = (data,)
        data = tuple(d.to(device) for d in data)
        if target is not None:
            target = target.to(device)
        # for name, param in model.named_parameters():
        #         if param.requires_grad:
        #             print(name,param)
        optimizer.zero_grad()
        if loss_fn.cross_entropy_flag:
            output_embedding, output_cross_entropy = model(*data)
            blended_loss, losses = loss_fn.calculate_loss(target, output_embedding, output_cross_entropy)
        else:
            output_embedding = model(*data)
            # print(output_embedding.shape,output_embedding)
            blended_loss, losses = loss_fn.calculate_loss(target, output_embedding)
        total_loss += blended_loss.item()
        blended_loss.backward()

        optimizer.step()

        # Print log
        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]'.format(
                batch_idx * len(data[0]), len(train_loader.dataset), 100. * batch_idx / len(train_loader))
            for name, value in losses.items():
                message += '\t{}: {:.6f}'.format(name, np.mean(value))
 
            print(message)

    total_loss /= (batch_idx + 1)
    return total_loss

def retrieve3(model, queries, cross_entropy_flag, train_dataset, img_size, infer_batch_size):

    query_img_dataset = test_data_generator(queries, img_size=img_size)
    query_loader = DataLoader(query_img_dataset, batch_size=infer_batch_size, shuffle=False, num_workers=4,
                              pin_memory=True)
    model.eval()
    model.cuda()
    query_paths, query_vecs = batch_process3(model, cross_entropy_flag, query_loader)
    # print(len(query_paths),len(query_vecs))
    num = 0
    for index, vec in enumerate(query_vecs):
        # print(vec)
        label = query_paths[index].split('_')[0]
        indice = train_dataset.class_to_idx[label]
        # print(indice,vec.argmax())
        if indice == vec.argmax():
            num += 1
    print('accuracy:', num/len(query_vecs))
    with open("./result.txt","a+") as f:
        f.writelines('accuracy:' + str(num/len(query_vecs))+ '\n')




