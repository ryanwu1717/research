import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist
from .model_utils import ActivationLayer, NormLayer, PositionalEncoding, gradient
from .NeRV import NeRV_MLP, NeRVBlock, Conv_Up_Block
from einops import rearrange
import tinycudann as tcnn
import modulation
from sparsegrid import SparseGrid

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not(heads==1 and dim_head==dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., prenorm=False):
        super(TransformerBlock, self).__init__()
        if prenorm:
            self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
            self.ffn = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        else:
            self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ffn(x) + x
        return x


class E_NeRV_Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # t mapping
        self.pe_t = PositionalEncoding(
            pe_embed_b=cfg['pos_b'], pe_embed_l=cfg['pos_l']
        )

        stem_dim_list = [int(x) for x in cfg['stem_dim_num'].split('_')]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in cfg['fc_hw_dim'].split('_')]
        self.block_dim = cfg['block_dim']

        mlp_dim_list = [self.pe_t.embed_length] + stem_dim_list + [144]
        self.stem_t = NeRV_MLP(dim_list=mlp_dim_list, act=cfg['act'])




        # xy mapping
        xy_coord = torch.stack( 
            torch.meshgrid(
                torch.arange(self.fc_h) / self.fc_h, torch.arange(self.fc_w) / self.fc_w
            ), dim=0
        ).flatten(1, 2)  # [2, h*w]
        self.xy_coord = nn.Parameter(xy_coord, requires_grad=False)
        self.pe_xy = PositionalEncoding(
            pe_embed_b=cfg['xypos_b'], pe_embed_l=cfg['xypos_l']
        )
        self.stem_xy = NeRV_MLP(dim_list=[2 * self.pe_xy.embed_length, self.block_dim ], act=cfg['act'])
        # self.stem_xy = NeRV_MLP(dim_list=[cfg['2d_encoding_xy']['n_levels'] * cfg['2d_encoding_xy']['n_features_per_level'], self.block_dim], act=cfg['act'])
        self.trans1 = TransformerBlock(
            dim=self.block_dim  , heads=1, dim_head=64, mlp_dim=cfg['mlp_dim'], dropout=0., prenorm=False
        )
        self.trans2 = TransformerBlock(
            dim=self.block_dim, heads=8, dim_head=64, mlp_dim=cfg['mlp_dim'], dropout=0., prenorm=False
        )
        if self.block_dim == self.fc_dim:
            self.toconv = nn.Identity()
        else:
            self.toconv = NeRV_MLP(dim_list=[self.block_dim, self.fc_dim], act=cfg['act'])
        
        # BUILD CONV LAYERS
        self.layers, self.head_layers, self.t_layers, self.norm_layers = [nn.ModuleList() for _ in range(4)]
        ngf = self.fc_dim
        for i, stride in enumerate(cfg['stride_list']):
            if i == 0:
                # expand channel width at first stage
                new_ngf = int(ngf * cfg['expansion'])
            else:
                # change the channel width for each stage
                new_ngf = max(ngf // (1 if stride == 1 else cfg['reduction']), cfg['lower_width'])
            
            self.t_layers.append(NeRV_MLP(dim_list=[128, 2*ngf], act=cfg['act']))
            self.norm_layers.append(nn.InstanceNorm2d(ngf, affine=False))
            
            if i == 0:
                self.layers.append(Conv_Up_Block(ngf=ngf, new_ngf=new_ngf, stride=stride, bias=cfg['bias'], norm=cfg['norm'], act=cfg['act'], conv_type=cfg['conv_type']))
            else:
                self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=stride, bias=cfg['bias'], norm=cfg['norm'], act=cfg['act'], conv_type=cfg['conv_type']))
            ngf = new_ngf

            # build head classifier, upscale feature layer, upscale img layer 
            head_layer = [None]
            if cfg['sin_res']:
                if i == len(cfg['stride_list']) - 1:
                    head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=cfg['bias'])
                else:
                    head_layer = None
            else:
                head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=cfg['bias'])
            self.head_layers.append(head_layer)
        self.sigmoid = cfg['sigmoid']

        self.T_num = 20
        self.pe_t_manipulate = PositionalEncoding(pe_embed_b=cfg['pos_b_tm'], pe_embed_l=cfg['pos_l_tm'])
        self.t_branch = NeRV_MLP(dim_list=[self.pe_t_manipulate.embed_length, 128, 128], act=cfg['act'])

        self.loss = cfg['additional_loss'] if cfg.__contains__('additional_loss') else None
        self.loss_w = cfg['additional_loss_weight'] if cfg.__contains__('additional_loss_weight') else 1.0
        self.mse = nn.MSELoss()

        # self.test = torch.nn.Conv1d(1024, 512, 3, stride=1,padding=1)
        # # self.test = torch.nn.Conv1d(2048, 144, 3, stride=1,padding=1)
        # self.test1 = torch.nn.Conv1d(512, 256, 3, stride=1,padding=1)
        # self.test2 = torch.nn.Conv1d(256, 144, 3, stride=1,padding=1)

        # self.test  = nn.Sequential(
        #         torch.nn.Conv1d(600, 200, 3, stride=1,padding=1),
        #         torch.nn.ReLU(),
        #         torch.nn.Conv1d(200, 144, 3, stride=1,padding=1),
        #         torch.nn.ReLU()
        #         )
        
        #add 
        # learnable keyframes xy
        # self.keyframes_xy = tcnn.Encoding(n_input_dims=2, encoding_config=cfg["2d_encoding_xy"])    
        # assert self.keyframes_xy.dtype == torch.float32
        # # learnable keyframes yt
        # self.keyframes_yt = tcnn.Encoding(n_input_dims=2, encoding_config=cfg["2d_encoding_yt"]) # torch.Tensor       
        # assert self.keyframes_yt.dtype == torch.float32

        # # learnable keyframes xt
        # self.keyframes_xt = tcnn.Encoding(n_input_dims=2, encoding_config=cfg["2d_encoding_xt"]) # torch.Tensor     
        # assert self.keyframes_xt.dtype == torch.float32

        out_features = 196
        self.net = modulation.SirenNet(
                                    dim_in = 1, # input dimension, ex. 2d coor
                                    dim_hidden = cfg["network"]["n_neurons"],       # hidden dimension
                                    dim_out = out_features,                                     # output dimension, ex. rgb value
                                    num_layers = cfg["network"]["n_hidden_layers"], # number of layers
                                    w0_initial = 30.,                                            # different signals may require 
                                                                                                 # different omega_0 in the first layer                                                               #  - this is a hyperparameter
                                    )
                                    
        # self.sparse_grid = SparseGrid(level_dim=cfg["3d_encoding"]["n_features_per_level"], 
        #                             x_resolution=cfg["3d_encoding"]["x_resolution"],
        #                             y_resolution=cfg["3d_encoding"]["y_resolution"],
        #                             t_resolution=cfg["3d_encoding"]["t_resolution"], 
        #                             upsample=cfg["3d_encoding"]["upsample"]
        #                             )
        # latent_dim = cfg["2d_encoding_xy"]["n_levels"]*(cfg["2d_encoding_xy"]["n_features_per_level"])
        # latent_dim += cfg["2d_encoding_yt"]["n_levels"]*(cfg["2d_encoding_yt"]["n_features_per_level"])
        # latent_dim += cfg["2d_encoding_xt"]["n_levels"]*(cfg["2d_encoding_xt"]["n_features_per_level"])
        # latent_dim += (cfg["3d_encoding"]["n_features_per_level"])*9

        self.wrapper = modulation.SirenWrapper(self.net, latent_dim = 256)

    def fuse_t(self, x, t):
        # x: [B, C, H, W], normalized among C
        # t: [B, 2* C]
        f_dim = t.shape[-1] // 2
        gamma = t[:, :f_dim]
        beta = t[:, f_dim:]

        gamma = gamma[..., None, None]
        beta = beta[..., None, None]
        out = x * gamma + beta
        return out

    def forward_impl(self, input_id , all_coords , timesteps):

        b, t = timesteps.size(0), timesteps.size(1)
        timesteps = timesteps.reshape(b*t, -1)

        all_coords = all_coords.view(-1, 3) # t, x, y
        t = input_id

        t_emb = self.stem_t(self.pe_t(t)) # [B, L]
        t_manipulate = self.t_branch(self.pe_t_manipulate(t))

        xy_coord = self.xy_coord
        x_coord = self.pe_xy(xy_coord[0])    # [h*w, C]
        y_coord = self.pe_xy(xy_coord[1])    # [h*w, C]
        xy_emb = torch.cat([x_coord, y_coord], dim=1)

        # xy_emb = self.stem_xy(xy_emb).unsqueeze(0).expand(t_emb.shape[0], -1, -1)  # [B, h*w, L]
        xy_emb = self.stem_xy(xy_emb).expand(t_emb.shape[0], -1, -1).squeeze(0)  # [B, h*w, L]
        # print("xy_emb",xy_emb.shape)
        # xy_emb = self.trans1(xy_emb).squeeze(0)
        # print("xy_emb",xy_emb.shape)
        # add
        # tmpxy = all_coords[:, [1, 2]]
        # xt_coords = all_coords[:, [0, 1]]
        # yt_coords = all_coords[:, [0, 2]]

        # spatial_embedding_xy = self.keyframes_xy(tmpxy)
        # spatial_embedding_xt = self.keyframes_xt(xt_coords)
        # spatial_embedding_yt = self.keyframes_yt(yt_coords) 
        
        # spatial_embedding = torch.cat((spatial_embedding_xy, spatial_embedding_yt, spatial_embedding_xt), dim=1)
        # motion_embedding = self.sparse_grid(all_coords)


        # embedding = torch.cat((spatial_embedding, motion_embedding), dim=1)
        # print("embedding",embedding.shape)

        # permute_xy = tmpxy.permute(1,0)

        # xy_coords = self.pe_xy(all_coords[:, [1, 2]])
        # # permute_xy = xy_coords.permute(1,0)
        # # print("QQ",all_coords[:, [1, 2]].shape)
        # # print("permute_xy",permute_xy[0].shape)
        # # print("xy_coords",xy_coords.shape)
        # x_coord = self.pe_xy(permute_xy[0])    # [h*w, C]
        # y_coord = self.pe_xy(permute_xy[1])    # [h*w, C]

        
        # print("xy_coord",xy_coord[0].shape)
        # print("x_coord",x_coord.shape)
        # print("y_coord",y_coord.shape)
        # xy_emb = torch.cat([x_coord, y_coord], dim=1)
        # print("xy_emb",xy_emb.shape)

        # xy_emb = self.testLinear(spatial_embedding_xy)
        # xy_emb = spatial_embedding_xy
        # print("stem_xy,xy_emb , t_emb",self.stem_xy(xy_emb).shape,xy_emb.shape ,t_emb.shape)
        # print("xy_emb ",spatial_embedding_xy.shape )
        # embedding = spatial_embedding_xy.squeeze(0)
        # spatial_embedding = torch.cat((spatial_embedding_xy, spatial_embedding_yt, spatial_embedding_xt), dim=1)
        # print("timesteps",timesteps.shape)
        # 
        emb = self.wrapper(coords=t_emb.permute(1,0), latent=xy_emb).unsqueeze(0)
        # xy_emb = self.test(xy_emb)
        # xy_emb = self.test1(xy_emb)
        # print("tmpoutput",xy_emb.shape)
        #original
        # print("unsqueeze ",self.stem_xy(xy_emb).unsqueeze(0).shape )
        # xy_emb = self.stem_xy(xy_emb).unsqueeze(0).expand(t_emb.shape[0], -1, -1)  # [B, h*w, L]
        # print("newxy_emb ",xy_emb.shape )
        
        # xy_emb = self.trans1(xy_emb)

        # fuse t into xy map
        # t_emb_list = [t_emb for i in range(xy_emb.shape[1])]
        # t_emb_map = torch.stack(t_emb_list, dim=1)  # [B, h*w, L]
        # emb = xy_emb * t_emb_map

        # print("tmp",self.trans2(emb).shape)

        # emb = self.toconv(self.trans2(emb))
        # print("emb",emb.shape)


        emb = emb.reshape(emb.shape[0], self.fc_h, self.fc_w, emb.shape[-1])
        # print(emb.shape)
        emb = emb.permute(0, 3, 1, 2)
        output = emb

        out_list = []
        for layer, head_layer, t_layer, norm_layer in zip(self.layers, self.head_layers, self.t_layers, self.norm_layers):
            # t_manipulate
            output = norm_layer(output)
            t_feat = t_layer(t_manipulate)
            output = self.fuse_t(output, t_feat)
            # conv
            output = layer(output) 
            if head_layer is not None:
                img_out = head_layer(output)
                # normalize the final output iwth sigmoid or tanh function
                img_out = torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5
                out_list.append(img_out)

        return  out_list
    
    def forward(self, data):
        input_id = data['img_id']  # [B]
        all_coords = data['all_coords']
        timesteps = data['temporal_steps']

        # print("all_coords",data['all_coords'].size())   
        batch_size = input_id.shape[0]
        

        output_list = self.forward_impl(input_id , all_coords , timesteps)  # a list containing [B or 2B, 3, H, W]

        if self.loss and self.training:
            b, c, h, w = output_list[-1].shape
            # NO USE
            grad_loss = 0.0
            return {
                "loss": grad_loss * self.loss_w,
                "output_list": output_list,
            }
        
        return output_list
