from matplotlib import pyplot as plt
import numpy as np

class VisRecorder:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VisRecorder, cls).__new__(cls, *args, **kwargs)
            cls._instance.seq = None
            cls._instance.description = None
        return cls._instance
    
    def set_seq(self, value):
        self.seq = value
    
    def set_description(self, value):
        self.description = value
    
    def get_filename(self):
        return f"se{self.seq}_{self.description}"


def my_mask_vis(x, save_dir, filename, figsize=(14, 3.8)):
    if len(x.shape) == 3:
        assert x.shape[0] == 1
        x = x[0]
    else:
        assert len(x.shape) == 2
    H, W = x.shape
    x_norm = x.detach().cpu() if x.requires_grad else x.cpu()
    
    alpha_mask1 = np.zeros((H, W, 4), dtype=float)
    # alpha_mask1[:, :, 0] = x_norm
    alpha_mask1[:, :, 2] = x_norm
    alpha_mask1[:, :, 3] = x_norm
    fig = plt.figure(figsize=figsize)
    plt.imshow(alpha_mask1)
    # im = plt.imshow(x_norm, cmap='Blues')
    # cbar = fig.colorbar(im, extend='both', shrink=0.9)
    # cbar.set_label('value')
    plt.savefig(f"{save_dir}/blue_{filename}_mask.jpg", format="jpg", dpi=500)
    plt.clf()
    
    plt.close("all")

def vis_bev_mask(x, save_dir, filename, figsize=(14, 3.8)):
    if len(x.shape) == 3:
        assert x.shape[0] == 1
        x = x[0]
    else:
        assert len(x.shape) == 2
    H, W = x.shape
    x_norm = x.detach().cpu() if x.requires_grad else x.cpu()
    fig = plt.figure(figsize=figsize)
    im = plt.imshow(x_norm, cmap='Blues')
    cbar = fig.colorbar(im, extend='both', shrink=0.9)
    cbar.set_label('value')
    plt.savefig(f"{save_dir}/{filename}_mask.jpg", format="jpg", dpi=500)
    plt.clf()
    
    plt.close("all")


def vis_bev_feature(x, save_dir, filename, vis_max=True, vis_avg=True, vis_channel=(8, 8), begin_channel=0, figsize=(15, 15), dpi=200):
    assert len(x.shape) == 3
    assert len(vis_channel) == 2
    C, H, W = x.shape
    
    x_cpu = x.detach().cpu() if x.requires_grad else x.cpu()
    
    xmin = (x_cpu.view(C, -1).min(dim=1).values).view(C, 1, 1)
    xmax = (x_cpu.view(C, -1).max(dim=1).values).view(C, 1, 1)
    x_norm = (x_cpu - xmin) / (xmax - xmin)
    
    x_norm = x_cpu
    fig = plt.figure(figsize=figsize)
    for i in range(vis_channel[0]):
        for j in range(vis_channel[1]):
            now_channel = begin_channel + i * vis_channel[1] + j
            ax = plt.subplot(vis_channel[0], vis_channel[1], now_channel + 1)
            im = plt.imshow(x_norm[now_channel, :, :])
            if i == vis_channel[0] - 1 and j == vis_channel[1] - 1:
                cbar = fig.colorbar(im, extend='both', spacing='proportional', shrink=0.8, ax=ax)
                cbar.set_label('value')
    plt.savefig(f"{save_dir}/{filename}.jpg", format="jpg", dpi=dpi)
    plt.clf()
    
    if vis_max:
        max_f = x_norm.max(dim=0).values
        im = plt.imshow(max_f)
        cbar = fig.colorbar(im, extend='both', spacing='proportional', shrink=0.8)
        cbar.set_label('value')
        plt.savefig(f"{save_dir}/{filename}_max.jpg", format="jpg", dpi=dpi)
        plt.clf()
    
    if vis_avg:
        avg_f = x_norm.mean(dim=0)
        im = plt.imshow(avg_f)
        cbar = fig.colorbar(im, extend='both', spacing='proportional', shrink=0.8)
        cbar.set_label('value')
        plt.savefig(f"{save_dir}/{filename}_avg.jpg", format="jpg", dpi=dpi)
        plt.clf()
    
    plt.close("all")

