# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from math import sin, cos, pi  # rotating regions
from math import floor  # truncating naics codes
import numba  # speed up data transform with JIT compilation


dataroot = 'neural-net/'
outputroot = 'neural-net/'

use_saved_model = False
saved_model_filename = ''

"""Define Constants"""

# "grid coordinates" created in R are in meters
cell_width = 0.025 * 1.609344 * 1000  # cell width in meters (convert from miles)
size_potential = 10  # potential locations: num_width_potential x num_width_potential
size_padding = 20  # number of padding cells on each side of potential grid
nc = 9  # number of channels: 1) other grocery stores 2) other businesses
BATCH_SIZE_real = 32  # regions with missing grocery store per batch
BATCH_SIZE_fill = 16  # regions with real location filled in (-> no missing) per batch
BATCH_SIZE_random = 16  # random regions (-> no missing) per batch
BATCH_SIZE = BATCH_SIZE_real + BATCH_SIZE_fill + BATCH_SIZE_random

frac_train_real = 1  # fraction of real regions to use for training
frac_train_random = 1  # fraction of random (unrealized) regions to use for training

use_cuda = True

curr_epoch = 0
epoch_set_seed = list()
epoch_set_seed.append(curr_epoch)
EPOCHS = 20
ITERS = 10000

print('BATCH_SIZE: ' + str(BATCH_SIZE))
print('cell_width: ' + str(round(cell_width)) + 'm')
# print('prob_none: ' + str(round(100 * prob_none)) + '%')

"""Read in data"""

dict_S_I = dict()
dict_S_I_restaurant = dict()
dict_S_I_recreation = dict()
dict_S_I_religious = dict()
dict_S_I_museum = dict()
dict_S_I_school = dict()
dict_S_I_daycare = dict()
dict_S_I_gas = dict()

print('reading in data')

# read in data of businesses near each grocery store
with open(dataroot+'grid_S_I.csv','r') as f:
    for line in f:
        # skip header
        if line.startswith('s_id'):
            continue
        # extract data
        slist = line.strip().split(',')
        s_id = int(slist[0])
        i_id = int(slist[1])
        x = float(slist[2])
        y = float(slist[3])
        if slist[4] == 'NA':
            naics = -1
        else:
            naics = int(slist[4])
        tup = (x,y,naics)
        # create entry if first time we encounter s_id
        if not s_id in dict_S_I.keys():
            dict_S_I[s_id] = list()
        # add data to this s_id
        dict_S_I[s_id].append(tup)
        if floor(naics / 100) == 7225:
            # create entry if first time we encounter s_id
            if not s_id in dict_S_I_restaurant.keys():
                dict_S_I_restaurant[s_id] = list()
            # add data to this s_id
            dict_S_I_restaurant[s_id].append(tup)
        if floor(naics / 100) == 7139:
            # create entry if first time we encounter s_id
            if not s_id in dict_S_I_recreation.keys():
                dict_S_I_recreation[s_id] = list()
            # add data to this s_id
            dict_S_I_recreation[s_id].append(tup)
        if floor(naics / 100) == 8131:
            # create entry if first time we encounter s_id
            if not s_id in dict_S_I_religious.keys():
                dict_S_I_religious[s_id] = list()
            # add data to this s_id
            dict_S_I_religious[s_id].append(tup)
        if floor(naics / 100) == 7121:
            # create entry if first time we encounter s_id
            if not s_id in dict_S_I_museum.keys():
                dict_S_I_museum[s_id] = list()
            # add data to this s_id
            dict_S_I_museum[s_id].append(tup)
        if floor(naics / 100) == 6111:
            # create entry if first time we encounter s_id
            if not s_id in dict_S_I_school.keys():
                dict_S_I_school[s_id] = list()
            # add data to this s_id
            dict_S_I_school[s_id].append(tup)
        if floor(naics / 100) == 6244:
            # create entry if first time we encounter s_id
            if not s_id in dict_S_I_daycare.keys():
                dict_S_I_daycare[s_id] = list()
            # add data to this s_id
            dict_S_I_daycare[s_id].append(tup)
        if floor(naics / 100) == 4471:
            # create entry if first time we encounter s_id
            if not s_id in dict_S_I_gas.keys():
                dict_S_I_gas[s_id] = list()
            # add data to this s_id
            dict_S_I_gas[s_id].append(tup)

# read in data of businesses near each grocery store
with open(dataroot+'grid_S_random_I.csv','r') as f:
    for line in f:
        # skip header
        if line.startswith('s_id'):
            continue
        # extract data
        slist = line.strip().split(',')
        s_id = int(slist[0])
        i_id = int(slist[1])
        x = float(slist[2])
        y = float(slist[3])
        if slist[4] == 'NA':
            naics = -1
        else:
            naics = int(slist[4])
        tup = (x,y,naics)
        # create entry if first time we encounter s_id
        if not s_id in dict_S_I.keys():
            dict_S_I[s_id] = list()
        # add data to this s_id
        dict_S_I[s_id].append(tup)
        if floor(naics / 100) == 7225:
            # create entry if first time we encounter s_id
            if not s_id in dict_S_I_restaurant.keys():
                dict_S_I_restaurant[s_id] = list()
            # add data to this s_id
            dict_S_I_restaurant[s_id].append(tup)
        if floor(naics / 100) == 7139:
            # create entry if first time we encounter s_id
            if not s_id in dict_S_I_recreation.keys():
                dict_S_I_recreation[s_id] = list()
            # add data to this s_id
            dict_S_I_recreation[s_id].append(tup)
        if floor(naics / 100) == 8131:
            # create entry if first time we encounter s_id
            if not s_id in dict_S_I_religious.keys():
                dict_S_I_religious[s_id] = list()
            # add data to this s_id
            dict_S_I_religious[s_id].append(tup)
        if floor(naics / 100) == 7121:
            # create entry if first time we encounter s_id
            if not s_id in dict_S_I_museum.keys():
                dict_S_I_museum[s_id] = list()
            # add data to this s_id
            dict_S_I_museum[s_id].append(tup)
        if floor(naics / 100) == 6111:
            # create entry if first time we encounter s_id
            if not s_id in dict_S_I_school.keys():
                dict_S_I_school[s_id] = list()
            # add data to this s_id
            dict_S_I_school[s_id].append(tup)
        if floor(naics / 100) == 6244:
            # create entry if first time we encounter s_id
            if not s_id in dict_S_I_daycare.keys():
                dict_S_I_daycare[s_id] = list()
            # add data to this s_id
            dict_S_I_daycare[s_id].append(tup)
        if floor(naics / 100) == 4471:
            # create entry if first time we encounter s_id
            if not s_id in dict_S_I_gas.keys():
                dict_S_I_gas[s_id] = list()
            # add data to this s_id
            dict_S_I_gas[s_id].append(tup)

dict_S_S = dict()
# read in data of grocery stores near each grocery store
with open(dataroot+'grid_S_S.csv','r') as f:
    for line in f:
        # skip header
        if line.startswith('s_id'):
            continue
        # extract data
        slist = line.strip().split(',')
        s_id = int(slist[0])
        s_id_oth = int(slist[1])
        x = float(slist[2])
        y = float(slist[3])
        tup = (x,y)
        # create entry if first time we encounter s_id
        if not s_id in dict_S_S.keys():
            dict_S_S[s_id] = list()
        # add data to this s_id
        dict_S_S[s_id].append(tup)

# read in data of grocery stores near each grocery store
with open(dataroot+'grid_S_random_S.csv','r') as f:
    for line in f:
        # skip header
        if line.startswith('s_id'):
            continue
        # extract data
        slist = line.strip().split(',')
        s_id = int(slist[0])
        s_id_oth = int(slist[1])
        x = float(slist[2])
        y = float(slist[3])
        tup = (x,y)
        # create entry if first time we encounter s_id
        if not s_id in dict_S_S.keys():
            dict_S_S[s_id] = list()
        # add data to this s_id
        dict_S_S[s_id].append(tup)

dict_S_I_mat = dict()
for key, value in dict_S_I.items():
    # create a numpy array
    mat = np.empty((len(value), 2))
    for i in range(len(value)):
        mat[i,0] = value[i][0]
        mat[i,1] = value[i][1]
    dict_S_I_mat[key] = mat

dict_S_I_restaurant_mat = dict()
for key, value in dict_S_I_restaurant.items():
    # create a numpy array
    mat = np.empty((len(value), 2))
    for i in range(len(value)):
        mat[i,0] = value[i][0]
        mat[i,1] = value[i][1]
    dict_S_I_restaurant_mat[key] = mat
# regions without restaurants:
for key in set(dict_S_I.keys()) - set(dict_S_I_restaurant.keys()):
    mat = np.empty((0,2))
    dict_S_I_restaurant_mat[key] = mat

dict_S_I_recreation_mat = dict()
for key, value in dict_S_I_recreation.items():
    # create a numpy array
    mat = np.empty((len(value), 2))
    for i in range(len(value)):
        mat[i,0] = value[i][0]
        mat[i,1] = value[i][1]
    dict_S_I_recreation_mat[key] = mat
# regions without recreation:
for key in set(dict_S_I.keys()) - set(dict_S_I_recreation.keys()):
    mat = np.empty((0,2))
    dict_S_I_recreation_mat[key] = mat

dict_S_I_religious_mat = dict()
for key, value in dict_S_I_religious.items():
    # create a numpy array
    mat = np.empty((len(value), 2))
    for i in range(len(value)):
        mat[i,0] = value[i][0]
        mat[i,1] = value[i][1]
    dict_S_I_religious_mat[key] = mat
# regions without religious:
for key in set(dict_S_I.keys()) - set(dict_S_I_religious.keys()):
    mat = np.empty((0,2))
    dict_S_I_religious_mat[key] = mat

dict_S_I_museum_mat = dict()
for key, value in dict_S_I_museum.items():
    # create a numpy array
    mat = np.empty((len(value), 2))
    for i in range(len(value)):
        mat[i,0] = value[i][0]
        mat[i,1] = value[i][1]
    dict_S_I_museum_mat[key] = mat
# regions without museum:
for key in set(dict_S_I.keys()) - set(dict_S_I_museum.keys()):
    mat = np.empty((0,2))
    dict_S_I_museum_mat[key] = mat

dict_S_I_school_mat = dict()
for key, value in dict_S_I_school.items():
    # create a numpy array
    mat = np.empty((len(value), 2))
    for i in range(len(value)):
        mat[i,0] = value[i][0]
        mat[i,1] = value[i][1]
    dict_S_I_school_mat[key] = mat
# regions without school:
for key in set(dict_S_I.keys()) - set(dict_S_I_school.keys()):
    mat = np.empty((0,2))
    dict_S_I_school_mat[key] = mat

dict_S_I_daycare_mat = dict()
for key, value in dict_S_I_daycare.items():
    # create a numpy array
    mat = np.empty((len(value), 2))
    for i in range(len(value)):
        mat[i,0] = value[i][0]
        mat[i,1] = value[i][1]
    dict_S_I_daycare_mat[key] = mat
# regions without daycare:
for key in set(dict_S_I.keys()) - set(dict_S_I_daycare.keys()):
    mat = np.empty((0,2))
    dict_S_I_daycare_mat[key] = mat

dict_S_I_gas_mat = dict()
for key, value in dict_S_I_gas.items():
    # create a numpy array
    mat = np.empty((len(value), 2))
    for i in range(len(value)):
        mat[i,0] = value[i][0]
        mat[i,1] = value[i][1]
    dict_S_I_gas_mat[key] = mat
# regions without gas:
for key in set(dict_S_I.keys()) - set(dict_S_I_gas.keys()):
    mat = np.empty((0,2))
    dict_S_I_gas_mat[key] = mat


dict_S_S_mat = dict()
for key, value in dict_S_S.items():
    # create a numpy array
    mat = np.empty((len(value), 2))
    for i in range(len(value)):
        mat[i,0] = value[i][0]
        mat[i,1] = value[i][1]
    dict_S_S_mat[key] = mat
# regions without grocery stores:
for key in set(dict_S_I.keys()) - set(dict_S_S.keys()):
    mat = np.empty((0,2))
    dict_S_S_mat[key] = mat


S_id_all = list(dict_S_I.keys())
# print(S_id_all)
S_id_real = [el for el in S_id_all if el<1000]
# print(S_id_real)
S_id_random = [el for el in S_id_all if el>=1000]
# print(S_id_random)

"""Transform Data to grid (functions)"""

@numba.jit(nopython=True)
def cnt_in_cell_mat(x,y):
    out = np.zeros((2*size_padding+size_potential,2*size_padding+size_potential))
    for i in range(len(x)):
        if min(x[i],y[i])>= 0 and max(x[i],y[i])<2*size_padding+size_potential:
            out[y[i],x[i]] += 1
    return out


@numba.jit(nopython=True)
def data_shift_rotate(mat,shift_x=0,shift_y=0,theta=0,mirror_var=1):
    # rotate by theta
    theta = theta * pi / 180
    if not theta == 0:
        x = cos(theta) * mat[:,0] - sin(theta) * mat[:,1]
        y = sin(theta) * mat[:,0] + cos(theta) * mat[:,1]
        # rot = np.array([[cos(theta),sin(theta)],[-sin(theta),cos(theta)]])
        # xy = mat@rot
        # x = xy[:,0]
        # y = xy[:,1]
    else:
        x = mat[:,0]
        y = mat[:,1]
    # mirror the region
    if mirror_var == 1 or mirror_var == -1:
        x = mirror_var * x
    # shift by shift_x, shift_y
    x = x+shift_x
    y = y+shift_y

    return x, y  


def data_to_grid(mat,shift_x=0,shift_y=0,theta=0,mirror_var=1):

    x, y = data_shift_rotate(mat,shift_x,shift_y,theta,mirror_var)

    # fit into cells
    x = np.around(x/cell_width + size_padding).astype(int)
    y = np.around(y/cell_width + size_padding).astype(int)

    return cnt_in_cell_mat(x,y)


# create tensor of the proper size (1 channel currently)
grid = torch.zeros(BATCH_SIZE,nc,2*size_padding+size_potential, 2*size_padding+size_potential) #, dtype=torch.double)
labels = torch.empty(BATCH_SIZE, dtype=torch.int64)


def create_batch(grid=grid,labels=labels,sample_ids_real=S_id_real, sample_ids_random=S_id_random, return_transf=False):

    grid = grid*0
    labels = labels*0

    if return_transf:
        transf = np.zeros(shape=(BATCH_SIZE,5))

    for b in range(BATCH_SIZE):

        if b < BATCH_SIZE_real + BATCH_SIZE_fill:
            s_id = np.random.choice(sample_ids_real)
        else:
            s_id = np.random.choice(sample_ids_random)

        # get the businesses near s_id
        mat_S = dict_S_S_mat[s_id]
        mat_I = dict_S_I_mat[s_id]
        mat_I_restaurant = dict_S_I_restaurant_mat[s_id]
        mat_I_recreation = dict_S_I_recreation_mat[s_id]
        mat_I_religious = dict_S_I_religious_mat[s_id]
        mat_I_museum = dict_S_I_museum_mat[s_id]
        mat_I_school = dict_S_I_school_mat[s_id]
        mat_I_daycare = dict_S_I_daycare_mat[s_id]
        mat_I_gas = dict_S_I_gas_mat[s_id]


        # randomly pick rotation of this region
        theta = np.random.rand()*360

        # randomly mirror? 
        mirror_var = (np.random.rand() > 0.5)*2 - 1

        # randomly pick where real store is going to be
        shift_x = np.random.rand()*cell_width*size_potential - cell_width/2
        shift_y = np.random.rand()*cell_width*size_potential - cell_width/2

        # print(shift_x,shift_y,theta,mirror_var)
        if return_transf:
            transf[b,0] = s_id
            transf[b,1] = shift_x
            transf[b,2] = shift_y
            transf[b,3] = theta
            transf[b,4] = mirror_var

        # fill tensor
        grid[b,0,:,:] = torch.from_numpy(data_to_grid(mat_S,shift_x=shift_x,shift_y=shift_y,theta=theta,mirror_var=mirror_var))
        if nc >= 2:
            grid[b,1,:,:] = torch.from_numpy(data_to_grid(mat_I,shift_x=shift_x,shift_y=shift_y,theta=theta,mirror_var=mirror_var))
        if nc >= 3:
            grid[b,2,:,:] = torch.from_numpy(data_to_grid(mat_I_restaurant,shift_x=shift_x,shift_y=shift_y,theta=theta,mirror_var=mirror_var))
        if nc >= 4:
            grid[b,3,:,:] = torch.from_numpy(data_to_grid(mat_I_recreation,shift_x=shift_x,shift_y=shift_y,theta=theta,mirror_var=mirror_var))
        if nc >= 5:
            grid[b,4,:,:] = torch.from_numpy(data_to_grid(mat_I_gas,shift_x=shift_x,shift_y=shift_y,theta=theta,mirror_var=mirror_var))
        if nc >= 6:
            grid[b,5,:,:] = torch.from_numpy(data_to_grid(mat_I_religious,shift_x=shift_x,shift_y=shift_y,theta=theta,mirror_var=mirror_var))
        if nc >= 7:
            grid[b,6,:,:] = torch.from_numpy(data_to_grid(mat_I_museum,shift_x=shift_x,shift_y=shift_y,theta=theta,mirror_var=mirror_var))
        if nc >= 8:
            grid[b,7,:,:] = torch.from_numpy(data_to_grid(mat_I_daycare,shift_x=shift_x,shift_y=shift_y,theta=theta,mirror_var=mirror_var))
        if nc >= 9:
            grid[b,8,:,:] = torch.from_numpy(data_to_grid(mat_I_school,shift_x=shift_x,shift_y=shift_y,theta=theta,mirror_var=mirror_var))

        # include this grocery store in the covariates?
        if b >= BATCH_SIZE_real and b < BATCH_SIZE_real+BATCH_SIZE_fill:
            treat_x = int(round(shift_x/cell_width) + size_padding)
            treat_y = int(round(shift_y/cell_width) + size_padding)
            grid[b,0,treat_y,treat_x] += 1

        # location of missing grocery store:
        if b < BATCH_SIZE_real:
            labels[b] = int(round(shift_y/cell_width)*size_potential) + int(round(shift_x/cell_width))
        # random region without missing grocery store or grocery store is filled in:
        else:
            labels[b] = pow(size_potential,2)  # index 1 larger than locations (start at 0)

    if not return_transf:
        return grid, labels
    else:
        return grid, labels, transf

"""Define Neural Net"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        main = nn.Sequential(
            nn.InstanceNorm2d(num_features=nc, affine=True),
            nn.Conv2d(in_channels=nc,
                out_channels=2*nc,
                kernel_size=5, #9,
                padding=2, #4, #(9-1)/2,
                padding_mode='replicate', # 'zeros', 'reflect' or 'replicate' could work
                bias=True),
            nn.InstanceNorm2d(num_features=2*nc, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=2*nc,
                out_channels=4*nc,
                kernel_size=21, #9,
                padding=20, #4, #(9-1)/2,
                padding_mode='replicate', # 'zeros', 'reflect' or 'replicate' could work
                dilation=2,
                bias=True),
            nn.InstanceNorm2d(num_features=4*nc, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4*nc,
                out_channels=4*nc,
                kernel_size=5, #9,
                padding=2, #4, #(9-1)/2,
                padding_mode='replicate', # 'zeros', 'reflect' or 'replicate' could work
                bias=True),
            nn.InstanceNorm2d(num_features=4*nc, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4*nc,
                out_channels=1,
                kernel_size=21, #9,
                padding=20, #4, #(9-1)/2,
                padding_mode='replicate', # 'zeros', 'reflect' or 'replicate' could work
                dilation=2,
                bias=True),
            nn.InstanceNorm2d(num_features=1, affine=True),
            nn.Flatten(),  #view(-1, 1*X_dim*X_dim),
            nn.Linear(1*pow(2*size_padding+size_potential,2), pow(size_potential,2)+1),  
        )
        self.main = main

    def forward(self, x):
        output = self.main(inputs)
        return output

"""function to initialize optimizer"""

def intitialize_optimizer(net):
    return optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

"""code for saving and loading the model"""

def save_model(filename=None):
    if not filename:
        date = (datetime.utcnow() + timedelta(hours=-7)).strftime('%Y-%m-%d--%H-%M')
        filename = 'checkpoint-epoch-' + str(curr_epoch) + '-' + date + '.tar'
    path_save = outputroot + filename
    # save the model
    torch.save({
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'curr_epoch': curr_epoch,
                'epoch_set_seed': epoch_set_seed,
                }, path_save)
    print('file: ' + path_save)

def load_model(filename,net=None,optimizer=None):
    global epoch_set_seed
    global curr_epoch
    path_load = outputroot + filename
    if not net:
        net = Net()
    # if using GPU
    if use_cuda and torch.cuda.is_available():
        net.cuda()
    if not optimizer:
        optimizer = intitialize_optimizer(net)
    if use_cuda and torch.cuda.is_available():
        print('CUDA available')
        checkpoint = torch.load(path_load)
    else:
        print('no CUDA...')
        device = torch.device('cpu')
        checkpoint = torch.load(path_load, map_location=device)
    net.load_state_dict(checkpoint['net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    curr_epoch = checkpoint['curr_epoch']
    if 'epoch_set_seed' in checkpoint.keys():
        epoch_set_seed = checkpoint['epoch_set_seed']
        print('found list in keys')
    epoch_set_seed.append(curr_epoch)



    return net, optimizer

"""Set random seed"""

# Set random seed for reproducibility
manualSeed = 24601
print('Random Seed: ', manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

"""Pick training / evaluation sample"""

# locations for training and evaluation
num_distinct_train_real = int(len(S_id_real) * frac_train_real)
num_distinct_train_random = int(len(S_id_random) * frac_train_random)

sample_train_real = list(np.random.choice(a=S_id_real,size=num_distinct_train_real,replace=False))
sample_train_real.sort()
sample_train_random = list(np.random.choice(a=S_id_random,size=num_distinct_train_random,replace=False))
sample_train_random.sort()

if num_distinct_train_real < len(S_id_real):
    sample_eval_real = list(set(S_id_real) - set(sample_train_real))
else:
    sample_eval_real = S_id_real
sample_eval_real.sort()
if num_distinct_train_random < len(S_id_random):
    sample_eval_random = list(set(S_id_random) - set(sample_train_random))
else:
    sample_eval_random = S_id_random
sample_eval_random.sort()

"""Initialize neural net"""

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if use_saved_model:
    print('Loading model')
    net, optimizer = load_model(saved_model_filename)
else:
    net = Net()
    net.apply(weights_init)
    if use_cuda and torch.cuda.is_available():
        net.cuda()
    optimizer = intitialize_optimizer(net)

criterion = nn.CrossEntropyLoss()

if use_cuda and torch.cuda.is_available():
    net.cuda()

# Set random seed for reproducibility: increment to ensure different training samples after load
manualSeed = 24601 + curr_epoch
#manualSeed = random.randint(1, 10000) # use if you want new results
print('Random Seed: ', manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

"""Training"""

print('Starting Training')
print((datetime.utcnow() + timedelta(hours=-7)).strftime('%Y-%m-%d %H:%M:%S'))

bound_epochs = curr_epoch + EPOCHS

# loop over the dataset multiple times
for epoch in range(curr_epoch, bound_epochs):

    # initialize fit statistics
    running_loss = 0.0
    correct = 0
    correct_real = 0
    non_zero_real = 0
    correct_fill = 0
    correct_random = 0
    total = 0
    total_real = 0
    total_fill = 0
    total_random = 0

    for i in range(ITERS):
        # get the inputs; data is a list of [inputs, labels]
        data = create_batch(sample_ids_real=sample_train_real,
                            sample_ids_random=sample_train_random)
        inputs, labels = data

        if use_cuda and torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # determine accuracy of taking "prediction"
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            correct_real += (predicted[0:BATCH_SIZE_real] == labels[0:BATCH_SIZE_real]).sum().item()
            non_zero_real += (predicted[0:BATCH_SIZE_real] < pow(size_potential,2)).sum().item()
            correct_fill += (predicted[BATCH_SIZE_real:BATCH_SIZE_real+BATCH_SIZE_fill] == labels[BATCH_SIZE_real:BATCH_SIZE_real+BATCH_SIZE_fill]).sum().item()
            correct_random += (predicted[BATCH_SIZE-BATCH_SIZE_random:BATCH_SIZE] == labels[BATCH_SIZE-BATCH_SIZE_random:BATCH_SIZE]).sum().item()
            total_real += BATCH_SIZE_real
            total_fill += BATCH_SIZE_fill
            total_random += BATCH_SIZE_random

        # print statistics
        running_loss += loss.item()
        if i % min(10000,ITERS/10) == min(10000,ITERS/10)-1:    # print every min(1000,ITERS/10) mini-batches
            print('[%d / %d, %5d / %5d] loss: %.3f, accuracy: %.1f%%, real: %.1f%%, real non-zero: %.1f%%, real filled: %.1f%%, unrealized: %.1f%%' %
                  (epoch + 1, bound_epochs, i + 1, ITERS, running_loss / min(10000,ITERS/10),
                   100 * correct / total,
                   100 * correct_real / max(total_real,1),
                   100 * non_zero_real / max(total_real,1),
                   100 * correct_fill / max(total_fill,1),
                   100 * correct_random / max(total_random,1)))
            print((datetime.utcnow() + timedelta(hours=-7)).strftime('%Y-%m-%d %H:%M:%S'))
            running_loss = 0.0
            correct = 0
            correct_real = 0
            non_zero_real = 0
            correct_fill = 0
            correct_random = 0
            total = 0
            total_real = 0
            total_fill = 0
            total_random = 0
            
        # evaluation sample:
        if frac_train_real < 1 or frac_train_random < 1:
            if i % min(10000,ITERS/5) == min(10000,ITERS/5)-1:    # print every min(1000,ITERS/10) mini-batches
                eval_correct = 0
                eval_correct_real = 0
                eval_non_zero_real = 0
                eval_correct_fill = 0
                eval_correct_random = 0
                eval_total = 0
                eval_total_real = 0
                eval_total_fill = 0
                eval_total_random = 0

                with torch.no_grad():
                    for i in range(100):
                        inputs, labels = create_batch(sample_ids_real=sample_eval_real,
                                                    sample_ids_random=sample_eval_random)

                        if use_cuda and torch.cuda.is_available():
                            inputs = inputs.cuda()
                            labels = labels.cuda()

                        outputs = net(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        eval_total += labels.size(0)
                        eval_correct += (predicted == labels).sum().item()
                        eval_correct_real += (predicted[0:BATCH_SIZE_real] == labels[0:BATCH_SIZE_real]).sum().item()
                        eval_non_zero_real += (predicted[0:BATCH_SIZE_real] < pow(size_potential,2)).sum().item()
                        eval_correct_fill += (predicted[BATCH_SIZE_real:BATCH_SIZE_real+BATCH_SIZE_fill] == labels[BATCH_SIZE_real:BATCH_SIZE_real+BATCH_SIZE_fill]).sum().item()
                        eval_correct_random += (predicted[BATCH_SIZE-BATCH_SIZE_random:BATCH_SIZE] == labels[BATCH_SIZE-BATCH_SIZE_random:BATCH_SIZE]).sum().item()
                        eval_total_real += BATCH_SIZE_real
                        eval_total_fill += BATCH_SIZE_fill
                        eval_total_random += BATCH_SIZE_random

                print('Accuracy on hold-out: %.1f%%, real: %.1f%%, real non-zero: %.1f%%, real filled: %.1f%%, unrealized: %.1f%%' %
                    (100 * eval_correct / max(eval_total,1),
                    100 * eval_correct_real / max(eval_total_real,1),
                    100 * eval_non_zero_real / max(eval_total_real,1),
                    100 * eval_correct_fill / max(eval_total_fill,1),
                    100 * eval_correct_random / max(eval_total_random,1)))


    print('Finished Epoch ' + str(epoch+1) + ' of ' + str(bound_epochs) + '. Saving model and optimizer checkpoint.')
    curr_epoch = curr_epoch + 1
    save_model()
    print((datetime.utcnow() + timedelta(hours=-7)).strftime('%Y-%m-%d %H:%M:%S'))

print('Finished Training')

