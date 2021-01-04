import shutil
import os
import glob

move_dataset = False
separate_dataset = True


# ====================separate dataset===================== #
# ========================================================= #

if separate_dataset:
    dataset_name = 'FFHQ_AnimeFaces256Cleaner'
    root_dir = dataA_dir = f'{os.getcwd()}/dataset/{dataset_name}/'
    dataA_dir = f'{root_dir}dataA/'
    dataB_dir = f'{root_dir}dataB/'
    trainA_dir = f'{root_dir}trainA/'
    trainB_dir = f'{root_dir}trainB/'
    testA_dir = f'{root_dir}testA/'
    testB_dir = f'{root_dir}testB/'

    dataA_list = glob.glob(os.path.join(dataA_dir, '*'))
    dataB_list = glob.glob(os.path.join(dataB_dir, '*'))

    if os.path.isdir(trainA_dir):
        shutil.rmtree(trainA_dir)
    if os.path.isdir(trainB_dir):
        shutil.rmtree(trainB_dir)
    if os.path.isdir(testA_dir):
        shutil.rmtree(testA_dir)
    if os.path.isdir(testB_dir):
        shutil.rmtree(testB_dir)

    os.mkdir(trainA_dir)
    os.mkdir(trainB_dir)  
    os.mkdir(testA_dir)
    os.mkdir(testB_dir)  

    len_dataset = min(len(dataA_list), len(dataB_list))
    len_train_data = round(len_dataset * 0.9)
    # for i, (dataA, dataB) in enumerate(zip(dataA_list, dataB_list)):
    for i in range(len_train_data):
        dataA_name = dataA_list[i].split('/')[-1]
        shutil.copyfile(dataA_list[i], os.path.join(trainA_dir, dataA_name))
        dataB_name = dataB_list[i].split('/')[-1]
        shutil.copyfile(dataB_list[i], os.path.join(trainB_dir, dataB_name))
        if (i+1) % 100 == 0:
            print(f'train count = {i+1}/{len_train_data}')
        if i+1 == len_train_data:
            break

    for i in range(len_train_data, len_dataset):
        dataA_name = dataA_list[i].split('/')[-1]
        shutil.copyfile(dataA_list[i], os.path.join(testA_dir, dataA_name))
        dataB_name = dataB_list[i].split('/')[-1]
        shutil.copyfile(dataB_list[i], os.path.join(testB_dir, dataB_name))
        if (i+1) % 100 == 0:
            print(f'test count = {i+1-len_train_data}/{len_dataset-len_train_data}')
        if (i+1) == len_dataset:
            break

    print(f'train data {len_train_data}, test data {len_dataset-len_train_data}, total {len_dataset}')





# ====================move dataset===================== #
# ===================================================== #

if move_dataset:
    dataset_name = 'FFHQ'
    dataset_dir = f'{os.getcwd()}/dataset/'
    move_dir = f'{os.getcwd()}/dataset/{dataset_name}/'
    data_list = glob.glob(os.path.join(dataset_dir, '*.png'))

    if os.path.isdir(move_dir) is False:
        os.mkdir(move_dir)

    data_list_len = len(data_list)
    for i, data in enumerate(data_list):
        data_name = data.split('/')[-1]
        shutil.move(data, os.path.join(move_dir, data_name))
        if i % 100 == 0:
            print(f'count = {i}/{data_list_len}')