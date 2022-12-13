import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import glob

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of 0 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform, train=True):
        self.main_dir = main_dir
        self.transform = transform
        frame_idx, self.frame_path = [], []
        accum_img_num = []
        all_imgs = os.listdir(main_dir)
        all_imgs.sort()

        num_frame = 0 
        for img_id in all_imgs:
            self.frame_path.append(img_id)
            frame_idx.append(num_frame)  # if 135 frames in total, this list will store 0, 1, 2, ..., 133, 134
            num_frame += 1          

        # import pdb; pdb.set_trace; from IPython import embed; embed()
        accum_img_num.append(num_frame)
        # the id for first frame is 0 and the id for last is 1
        self.frame_idx = []
        # for i in range(len(frame_idx)):
        #     x = frame_idx[i]
        #     self.frame_idx.append(float(x) / (len(frame_idx) - 1))
        self.accum_img_num = np.asfarray(accum_img_num)

        self.height = 720
        self.width = 1280

        #coords
        video_path = os.path.join(main_dir, "*.png")
        files = sorted(glob.glob(video_path))
        self.split_num = len(files)
        tmp_img = Image.open(files[0])
        tmp_img  = np.array(tmp_img)
        tmp_shape = tmp_img.shape

        self.vid = np.zeros((self.split_num, tmp_shape[0], tmp_shape[1], tmp_shape[2]), dtype=np.uint8)
        self.height = tmp_shape[0]
        self.width = tmp_shape[1]

        for idx, f in enumerate(files):
            img = Image.open(f)
            img = np.array(img)
            self.vid[idx] = img

            x = frame_idx[idx]
            self.frame_idx.append(float(x) / (len(frame_idx) - 1))

        self.shape = self.vid.shape[1:-1]
        
        self.nframes = self.vid.shape[0]
        self.channels = self.vid.shape[-1]

        self.sidelength = self.shape
        self.mgrid = get_mgrid(self.shape, dim=2) # [w * h, 3]

        data = torch.from_numpy(self.vid) 
        print("data",data.shape)
        self.data = data.view(self.nframes, -1, self.channels) # [ f, w * h, 3]
        # batch 
        self.N_samples = 1024 

        half_dt =  0.5 / self.nframes

        # modulation input
        self.temporal_steps = torch.linspace(half_dt, 1-half_dt, self.nframes )
        # temporal coords
        self.temporal_coords = torch.linspace(0, 1, self.nframes )

    def __len__(self):
        return len(self.frame_idx)

    def __getitem__(self, idx):
        valid_idx = int(idx)
        img_id = self.frame_path[valid_idx]
        img_name = os.path.join(self.main_dir, img_id)
        image = Image.open(img_name).convert("RGB")
        
        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height))

        tensor_image = self.transform(image)
        if tensor_image.size(1) > tensor_image.size(2):
            tensor_image = tensor_image.permute(0,2,1)
        frame_idx = torch.tensor(self.frame_idx[idx])
        
        #add
        
        temporal_coord_idx = torch.randint(0, self.data.shape[0], (self.N_samples,)) 
        spatial_coord_idx = torch.randint(0, self.data.shape[1], (self.N_samples,))
        data = self.data[temporal_coord_idx, spatial_coord_idx, :] 
        
        spatial_coords = self.mgrid[spatial_coord_idx, :] 
        temporal_coords = self.temporal_coords[temporal_coord_idx] 
        
        temporal_steps = self.temporal_steps[temporal_coord_idx]


        all_coords = torch.cat((temporal_coords.unsqueeze(1), spatial_coords), dim=1)
        
        data_dict = {
            "img_id": frame_idx,
            "img_gt": tensor_image,
            "all_coords": all_coords,
            "temporal_steps": temporal_steps
        }
        return data_dict