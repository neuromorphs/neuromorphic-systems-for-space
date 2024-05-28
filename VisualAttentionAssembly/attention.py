'''
Giulia D'Angelo
Postdoctoral Researcher
gvmdangelo@gmail.com

This code resides within a private repository exclusively accessible to Telluride.
Its primary function involves converting a video into frames,
generating events through a simple events generator,
and producing saliency maps for visual attention.

- video2frames: extracts frames, from the video and save them in the folder frames/.
    -frame_count=# of frames
    -fps=#frames per second
- frames2events: generates events from the frames creating a dict and saving the .pkl file.
    -thrsh_up (0.5,1] & thrsh_down [0,0.5) determine the sensitivity of the events generation. (for construction 0.5 cannot be chose)
- see_events: shows the previous generated events and save them (time window is set to be dur_frame = duration / frame_count).
- mk_videoframes: creates a video with the event frames.
- refactor_data: generates and save a npy array from the dict at the convenience of the attention model.
- run_attention: defines the parameters to run the attention
                                                    - fltr_resize_perc: percentage of resize for the Von Mises filter
                                                    - angle_shift: orientations of the Von Mises filter
                                                    - polarity: use single polarity
                                                    - num_pyr: # of scales for the pyramid (scale invariancy)
                                                    - show_imgs: flag to show the images
- attention: runs the attention model with a spiking CNN where there is a layer with Integrate and Fire neurons (IF).
  The model process one event frame at the time, where the neurons represent the Von Mises filter of different orientations.
  The different orientations are then summed up to obtain the saliency. The saliency map is then generates and saved.
  The model is a simplfied version and a follow-up of:
                                    D’Angelo, Giulia, et al. "Event driven bio-inspired attentive system for the iCub humanoid
                                    robot on SpiNNaker." Neuromorphic Computing and Engineering 2.2 (2022): 024008.
'''


import cv2
import sinabs.layers as sl
import torchvision
import random
import tonic
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import numpy.lib.recfunctions as rf
import torch.nn as nn
from skimage.transform import rescale, resize, downscale_local_mean
import glob
from sklearn import preprocessing as pre
from torchvision.utils import save_image


def norm_mat(mat):
    minv = [min(mat.min(0)), min(mat.min(1))]
    maxv = [max(mat.max(0)), max(mat.max(1))]
    mat = (mat - min(minv)) / (max(maxv) - min(minv))
    return mat

def video2frames(file, savepath):
    if os.path.isdir(savepath):
        print('frames dir already there!')
    else:
        os.mkdir(savepath)
    #load video
    video = cv2.VideoCapture(file)
    #get fps
    fps = video.get(cv2.CAP_PROP_FPS)
    print('frames per second =', fps)
    #number of frames
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('tot frames: ' + str(frame_count))
    #save frame count and fps
    with open('frame_count.pkl', 'wb') as f:
        pickle.dump(frame_count, f)
    with open('fps.pkl', 'wb') as f:
        pickle.dump(fps, f)
    for i in range(0,frame_count):
        print(str(i))
        # extract and save frames
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(savepath + str(i) + '.png', frame)
    return

def load_par():
    with open('frame_count.pkl', 'rb') as f:
        frame_count = pickle.load(f)
    with open('fps.pkl', 'rb') as f:
        fps = pickle.load(f)
    duration = frame_count / fps  # seconds
    dur_frame = duration / frame_count
    return frame_count,fps, dur_frame, duration

def frames2events(file, thrsh_up, thrsh_down):
    #loading frame count and fps
    frame_count,fps, dur_frame, duration = load_par()
    events = {'x': [], 'y': [], 'ts': [], 'pol': []}
    time_curr = 0
    timexfr = dur_frame
    print('tot frames: ' + str(frame_count))
    #generate events for each frame
    for idx in range(1,frame_count):
        print(str(idx))
        img_curr = Image.open(file+str(idx)+'.png')
        val_curr = np.array(img_curr.getdata())
        val_curr = val_curr.reshape((img_curr.size[1], img_curr.size[0]))

        img_past = Image.open(file + str(idx-1) + '.png')
        val_past = np.array(img_past.getdata())
        val_past = val_past.reshape((img_past.size[1], img_past.size[0]))

        #images difference & generate positive and negative polarities
        diff_imgs = val_curr - val_past
        diff_imgs = norm_mat(diff_imgs)
        diff=np.zeros((img_past.size[1], img_past.size[0]))
        diff[np.where(diff_imgs > thrsh_up)] = 1
        diff[np.where(diff_imgs == 0.0)] = 0.5
        diff[np.where(diff_imgs < thrsh_down)] = -1

        #create events
        y_pos, x_pos = np.where(diff == 1)
        y_neg, x_neg = np.where(diff == -1)

        tmp = {'x': [], 'y': [], 'ts': [], 'pol': []}
        tmp['x'][len(tmp['x']):] = np.concatenate((x_pos, x_neg), axis=0).tolist()
        tmp['y'][len(tmp['y']):] = np.concatenate((y_pos, y_neg), axis=0).tolist()
        tmp['pol'][len(tmp['pol']):] = np.concatenate((np.ones(len(x_pos)), np.zeros(len(x_neg))), axis=0).tolist()
        tmp['ts'][len(tmp['ts']):] = np.random.uniform(time_curr, (time_curr + timexfr),len(tmp['x']))

        #sort per ts and append events
        idxs = np.argsort(tmp['ts'])
        for eidx in idxs:
            events['x'].append(tmp['x'][eidx])
            events['y'].append(tmp['y'][eidx])
            events['ts'].append(tmp['ts'][eidx])
            events['pol'].append(tmp['pol'][eidx])
        time_curr = time_curr + timexfr
    with open('events.pkl', 'wb') as f:
        pickle.dump(events, f)
    return events

def see_events(file, savepath, wdw_dur):
    if os.path.isdir(savepath):
        print('frames dir already there!')
    else:
        os.mkdir(savepath)
    with open(file, 'rb') as f:
        events = pickle.load(f)
    w, h = [max(events['x'])+1, max(events['y'])+1]
    maxts=max(events['ts'])
    # checking events visualisation
    wdw = np.ones((h,w))*(255/2)
    wdt_per = wdw_dur
    ev_num=0
    print('tot ts: ' + str(maxts))
    for i in range(0, len(events['ts'])):
        if (events['pol'][i] == 1.0):
            wdw[events['y'][i],events['x'][i]] = 255
        if (events['pol'][i] == -1.0):
            wdw[events['y'][i], events['x'][i]] = 0
        if events['ts'][i] >= wdt_per:
            print(str(events['ts'][i]))
            plt.imshow(wdw,cmap='inferno')
            plt.axis('off')
            plt.draw()
            plt.pause(0.02)
            cv2.imwrite(savepath + str(ev_num) + '.png', wdw)
            wdw = np.zeros((h,w))
            wdt_per = wdt_per + wdw_dur
            ev_num+=1

def mk_videoframes(name_file,path_files):
    frame_count, fps, dur_frame, duration = load_par()
    img_array = []
    list_frames = os.listdir(path_files)
    for idx in range(len(list_frames)-1):
        print(str(idx))
        img = cv2.imread(path_files+str(idx)+'.png')
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(name_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    return


def attention(savepath, rec, angle_shift, fltr_resize_perc, time_wnd_frames, polarity, show_imgs, num_pyr):
    if os.path.isdir(savepath):
        print('frames dir already there!')
    else:
        os.mkdir(savepath)
    #orientations of the Von Mises (VM) filter
    angles = range(0, 360, angle_shift)
    filters = []
    for i in angles:
        filter = np.load(f"VMfilters/{i}_grad.npy")
        filter = rescale(filter, fltr_resize_perc, anti_aliasing=False)
        filters.append(filter)
    #tensor with 8 orientation VM filter
    filters = torch.tensor(np.stack(filters).astype(np.float32))
    print(f'VM filters size ({filter.shape[0]},{filter.shape[0]})')
    if show_imgs:
        #plt filters
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        fig.suptitle(f'VM filters size ({filter.shape[0]},{filter.shape[0]})', fontsize=16)
        #show filters
        for i in range(8):
            if i < 4:
                axes[0, i].set_title(f"{angles[i]} grad")
                axes[0, i].imshow(filters[i])
            else:
                axes[1, i-4].set_title(f"{angles[i]} grad")
                axes[1, i-4].imshow(filters[i])
    # define our single layer network and load the filters
    net = nn.Sequential(
        nn.Conv2d(1, filters.shape[0], filters.shape[1], bias=False),
        sl.IAF()
    )
    #define the weights of the network with the VM kernel of != orientations
    net[0].weight.data = filters.unsqueeze(1)
    # find out maximum x and y from the events
    max_x = rec['x'].max().astype(int)+1
    max_y = rec['y'].max().astype(int)+1
    max_ts = rec['t'].max()
    # use single polarity
    rec['p'] = polarity
    sensor_size = (max_x, max_y, 1)
    print(f"sensor size is {sensor_size}")
    # convert the raw events into frames so that we can feed those to our network depending on time_wnd_frames
    transforms = torchvision.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=time_wnd_frames),
        torch.tensor,
    ])
    frames = transforms(rec)
    for fr in range(len(frames)):
        # prepare the saliency map and the frame tensor
        salmap = torch.empty((1, num_pyr, max_y, max_x), dtype=torch.int64)
        frame = torch.empty((1, 1, max_y, max_x))
        frame[0,0, :, :] = frames[fr,0, :, :]
        # pyramidal scales to be scale invariant
        for pyr in range(1, num_pyr+1):
            print(f"pyramid scale {pyr}")
            res = (int((frame.shape[2])/pyr), int((frame.shape[3])/pyr))
            frames_ch = torchvision.transforms.Resize((res[0], res[1]))(frame)
            print(res)
            frames_ch.shape
            if show_imgs:
                plt.figure()
                plt.imshow(frames_ch[9, 0])
            # now we feed the data to our network!
            with torch.no_grad():
                output = net(frames_ch.float())
            output.shape
            # in the end we can plot the sal map
            if show_imgs:
                fig, axes = plt.subplots(2, 4, figsize=(10, 5))
                for i in range(8):
                    if i < 4:
                        axes[0, i].set_title(f"{angles[i]} grad")
                        axes[0, i].imshow(output[0, i])
                    else:
                        axes[1, i-4].set_title(f"{angles[i]} grad")
                        axes[1, i-4].imshow(output[0, i])
                plt.show()
            #sum over different rotations
            output_rotations = torch.sum(output, dim=1, keepdim=True)
            #sal map
            salmap[0,(pyr-1)] = torchvision.transforms.Resize((int(max_y),int(max_x)))(output_rotations[0])
        #sum over the pyramid
        salmap = torch.sum(salmap, dim=1, keepdim=True)
        img = salmap[0,0].detach().cpu().numpy()
        if show_imgs:
            plt.imshow(img, cmap='inferno')
            plt.axis('off')
            plt.draw()
            plt.pause(0.02)
        plt.imsave(savepath+str(fr) + '.png', img, cmap='inferno')
    return

def refactor_data(data):
    recording = np.load(data,allow_pickle=True)
    events=np.empty(shape=(len(recording['x']), 4))
    for idx in range(len(recording['x'])):
        print(str(idx))
        events[idx][0]=recording['x'][idx]
        events[idx][1]=recording['y'][idx]
        events[idx][2]=recording['pol'][idx]
        events[idx][3]=recording['ts'][idx]
    with open('events.npy', 'wb') as f:
        np.save(f, events)
    return


def run_attention(save_path):
    fltr_resize_perc = 2
    angle_shift = 45
    polarity = 0
    num_pyr = 6
    show_imgs = 0
    recording = np.load('events.npy')
    recording[:, 3] *= 1e6
    frame_count, fps, dur_frame, duration = load_par()
    time_wnd_frames = dur_frame * 1e6  # us
    rec = rf.unstructured_to_structured(recording,
                                        dtype=np.dtype(
                                            [('x', np.int16), ('y', np.int16), ('p', bool), ('t', int)]))
    attention(save_path, rec, angle_shift, fltr_resize_perc, time_wnd_frames, polarity, show_imgs, num_pyr)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ## create and save frames from video
    events=video2frames(file='Prophesee.mp4', savepath='frames/')

    ## generate events from frames, with intesity threashold thrsh_up, thrsh_down
    events=frames2events(file='frames/', thrsh_down=0.3, thrsh_up=0.7)

    ## visualise events, wdw_dur in seconds
    frame_count,fps, dur_frame, duration = load_par()
    see_events(file='events.pkl', savepath='eventframes/', wdw_dur=dur_frame)

    ## make video with event frames
    mk_videoframes(name_file='eventframes.mp4',path_files='eventframes/')

    # reformat events data to work with attention
    refactor_data(data='events.pkl')

    ## run attention model
    run_attention(save_path='attentionframes/')
    mk_videoframes(name_file = 'attention.mp4', path_files='attentionframes/')











