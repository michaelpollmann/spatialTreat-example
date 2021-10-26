from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from math import sin, cos, pi  # rotating regions
from math import floor  # truncating naics codes
import numba  # speed up data transform with JIT compilation

# Root directory for dataset
dataroot = 'neural-net/'
outputroot = 'neural-net/'

saved_model_filename = 'checkpoint-epoch-<epoch>-YYYY-MM-DD--hh-mm.tar'

"""Define Constants"""

# "grid coordinates" created in R are in meters
cell_width = 0.025 * 1.609344 * 1000  # cell width in meters (convert from miles)
size_potential = 10  # potential locations: num_width_potential x num_width_potential
size_padding = 20  # number of padding cells on each side of potential grid
nc = 9  # number of channels: 1) other grocery stores 2) other businesses

num_batches_predict = 5000
# change ratio of what regions to simulate since we don't care much about the real ones
BATCH_SIZE_real = 2  # regions with missing grocery store per batch
BATCH_SIZE_fill = 2  # regions with real location filled in (-> no missing) per batch
BATCH_SIZE_random = 28  # random regions (-> no missing) per batch
BATCH_SIZE = BATCH_SIZE_real + BATCH_SIZE_fill + BATCH_SIZE_random

use_cuda = True

"""Read in data"""

dict_S_I = dict()
dict_S_I_restaurant = dict()
dict_S_I_recreation = dict()
dict_S_I_religious = dict()
dict_S_I_museum = dict()
dict_S_I_school = dict()
dict_S_I_daycare = dict()
dict_S_I_gas = dict()

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

    # out = np.zeros((2*size_padding+size_potential,2*size_padding+size_potential))
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

"""Define neural nets"""
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

def initialize_optimizer(net):
    return optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

"""code for saving and loading the model"""

def load_model(filename,net=None,optimizer=None):
    global epoch_set_seed
    global curr_epoch
    path_load = dataroot + filename
    if not net:
        net = Net()
    # if using GPU
    if use_cuda and torch.cuda.is_available():
        net.cuda()
    if not optimizer:
        optimizer = initialize_optimizer(net)
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
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

"""load neural net"""

net, optimizer = load_model(saved_model_filename)

criterion = nn.CrossEntropyLoss()

if use_cuda and torch.cuda.is_available():
    net.cuda()

# Set random seed for reproducibility: increment to ensure different training samples after load
manualSeed = 24601 + curr_epoch
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

"""Functions to transform output into list for saving to file"""

@numba.jit
def data_reverse_shift_rotate(xy,shift_x=0,shift_y=0,theta=0,mirror_var=1):
    # reverse shift
    xy[:,0] -= shift_x
    xy[:,1] -= shift_y

    # reverse mirroring
    if mirror_var == 1 or mirror_var == -1:
        xy[:,0] = mirror_var * xy[:,0]
    
    # reverse rotation by theta
    theta = theta * pi / 180
    if not theta == 0:
        # x = cos(theta) * mat[:,0] - sin(theta) * mat[:,1]
        # y = sin(theta) * mat[:,0] + cos(theta) * mat[:,1]
        rot = np.array([[cos(theta),sin(theta)],[-sin(theta),cos(theta)]])
        xy = xy@np.linalg.inv(rot)
        # x = xy[:,0]
        # y = xy[:,1]
    return xy


def add_to_list(xy,o,r):
    for i in range(len(xy)):
        if i == len(xy) - 1:
            tup = (str(int(round(xy[i,0]))), str(r), 'NA', 'NA', str(o[i]))
        else:
            tup = (str(int(round(xy[i,0]))), str(r), str(int(round(xy[i,1]))), str(int(round(xy[i,2]))), str(o[i]))
        list_out.append(tup)


def outputs_to_loc(outputs,transf):
    o = outputs.cpu().numpy()
    g = np.linspace(start=cell_width/2,
                    stop=cell_width/2 + cell_width*size_potential,
                    num=size_potential, endpoint=False)
    
    for b in range(BATCH_SIZE):
        # grid cell midpoints
        xy = np.zeros(shape=(pow(size_potential,2)+1,3))
        # set s_id
        xy[:,0] = int(transf[b,0])
        # set relative location
        xy[0:pow(size_potential,2),1] = np.tile(g,size_potential)
        xy[0:pow(size_potential,2),2] = np.repeat(g,size_potential)

        xy[0:pow(size_potential,2),1:3] = data_reverse_shift_rotate(xy[0:pow(size_potential,2),1:3],
                                                                    shift_x=transf[b,1],
                                                                    shift_y=transf[b,2],
                                                                    theta=transf[b,3],
                                                                    mirror_var=transf[b,4])
        add_to_list(xy,o[b,:],b < BATCH_SIZE_real)

"""run the neural net on many practice examples to guess real locations"""

list_out = list()

with torch.no_grad():
    for i in range(num_batches_predict):
        # show progress
        if i % 100 == 0:
            print(str(i+1) + "/" + str(num_batches_predict) + " - time: " + (datetime.utcnow() + timedelta(hours=-7)).strftime('%Y-%m-%d %H:%M:%S'))

        # get the inputs; data is a list of [inputs, labels]
        data = create_batch(return_transf=True)
        inputs, labels, transf = data

        if use_cuda and torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        outputs_to_loc(outputs,transf)

# print(list_out)

print(len(list_out))

"""Save the resulting file

"""

date = (datetime.utcnow() + timedelta(hours=-7)).strftime('%Y-%m-%d--%H-%M')
filename_out = 'predicted_activation-' + date + '.csv'

with open(dataroot+filename_out,'w') as f:
    f.write('s_id,real_missing,x,y,activation\n')
    for e in list_out:
        f.write(','.join(e) + '\n')