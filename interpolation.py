import os
import torch
from torchvision import utils
from model2 import Generator

import glob
from PIL import Image
import sys
from tqdm import tqdm
import numpy as np

if not os.path.isdir('/sample/'):
    os.mkdir('sample')

if not os.path.isdir('/sg/'):
    os.mkdir('sg')

# need args:
# size
# sample
# npics
# checkpoint
# device

def gen(npics, G, device, seeds = [1], nsample = 1, styledim = 512, truncation = 1.0, trunc_mean = 4096):
  with torch.no_grad():
    G.eval()
    for i in tqdm(range(npics)):
      torch.manual_seed(seeds[i])
      sample_z = torch.randn(nsample, styledim, device = device)

      sample, _ = G(
          [sample_z], truncation = truncation, truncation_latent = trunc_mean
      )

      utils.save_image(
          sample,
          f"sample/{str(i).zfill(6)}.png",
          nrow = 1,
          normalize = True,
          range = (-1, 1),
      )

device = "cuda"
G = Generator(
    size = 128, style_dim = 512, n_mlp = 8
).to(device)

checkpoint = torch.load("/content/drive/MyDrive/style-based-gan-pytorch/checkpoints_corgi_reg_aug/040000.pt")

G.load_state_dict(checkpoint["g_ema"], strict = False)

n = 25000
gen(npics = n, G = G, device = "cuda", seeds = range(n))

# linear interpolation
def linterp(z, steps):
  out = []
  for i in range(len(z)-1):
    for index in range(steps):
      t = index/float(steps)
      out.append(z[i+1] * t + z[i] * (1-t))
  return out

# linear interpolation in z_space
def gen_linterp_z(G, device, nsteps = 5, seeds = [0, 2], styledim = 512, truncation = 1.0, trunc_mean = 4096):
  with torch.no_grad():
    G.eval()
    torch.manual_seed(seeds[0])
    start = torch.randn(1, styledim, device = device)
    torch.manual_seed(seeds[1])
    end = torch.randn(1, styledim, device = device)

    zs = linterp([start, end], steps = nsteps)

    for i in tqdm(range(nsteps)):

      sample, _ = G(
          [zs[i]], truncation = truncation, truncation_latent = trunc_mean
      )

      utils.save_image(
          sample,
          f"sample/{str(i).zfill(4)}.png",
          nrow = nsteps,
          normalize = True,
          range = (-1, 1),
      )

gen_linterp_z(G = G, device = "cuda", nsteps = 25)

# filepaths
fp_in = "/sg/sample/*.png"
fp_out = "/sg/sample/linterp.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=200, loop=0)

z = start = torch.randn(1, 512, device = device)
w = G.style(z)
z[0,0], w[0,0]

smp, _ = G(w, input_is_latent = True)
utils.save_image(
          smp,
          f"sample/test.png",
          nrow = 1,
          normalize = True,
          range = (-1, 1),
      )

# linear interpolation in w_space
# G(ws, input_is_latent = True)
def gen_linterp_w(G, device, nsteps = 5, seeds = [0, 2], styledim = 512, truncation = 1.0, trunc_mean = 4096):
  with torch.no_grad():
    G.eval()
    torch.manual_seed(seeds[0])
    start = torch.randn(1, styledim, device = device)
    torch.manual_seed(seeds[1])
    end = torch.randn(1, styledim, device = device)

    # pass through style network
    start_w = G.style(start)
    end_w = G.style(end)

    ws = linterp([start_w, end_w], steps = nsteps)

    for i in tqdm(range(nsteps)):

      sample, _ = G(
          [ws[i]],
          truncation = truncation,
          truncation_latent = trunc_mean,
          input_is_latent = True
      )

      utils.save_image(
          sample,
          f"sample_w/{str(i).zfill(4)}.png",
          nrow = nsteps,
          normalize = True,
          range = (-1, 1),
      )
if not os.path.isdir("/sg/sample_w/"):
    os.mkdir("/sg/sample_w/")

gen_linterp_w(G = G, device = "cuda", nsteps = 25)

# generate gif
fp_in = "/content/sg/sample_w/*.png"
fp_out = "/content/sg/sample_w/linterp_w.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=200, loop=0)

torch.manual_seed(0)
start = torch.randn(1, 2, device = "cpu")
torch.manual_seed(2)
end = torch.randn(1, 2, device = "cpu")

# spherical interpolation
def spherical_interp(steps, start, end):
  out = []
  for i in range(steps):
    t = i / (steps - 1)
    if t <= 0:
      out.append(start)
    elif t >= 1:
      out.append(end)
    elif torch.allclose(start, end):
      out.append(start)
    omega = torch.arccos(torch.tensordot(start/torch.linalg.norm(start), end/torch.linalg.norm(end)))
    sin_omega = torch.sin(omega)
    out.append(np.sin((1.0 - t) * omega) / sin_omega * start + torch.sin(t * omega) / sin_omega * end)
  return out

# spherical interpolation in z_space
def gen_slerp_z(G, device, nsteps = 5, seeds = [0, 2], styledim = 512, truncation = 1.0, trunc_mean = 4096):
  with torch.no_grad():
    G.eval()
    torch.manual_seed(seeds[0])
    start = torch.randn(1, styledim, device = device)
    torch.manual_seed(seeds[1])
    end = torch.randn(1, styledim, device = device)

    zs = spherical_interp(steps = nsteps, start = start.cpu(), end = end.cpu())
    zs = torch.stack(zs)
    zs = zs.to(torch.device('cuda'))

    for i in tqdm(range(nsteps)):

      sample, _ = G(
          [zs[i]], truncation = truncation, truncation_latent = trunc_mean
      )

      utils.save_image(
          sample,
          f"sample_spherical/{str(i).zfill(4)}.png",
          nrow = nsteps,
          normalize = True,
          range = (-1, 1),
      )

# spherical interpolation in w_space
# G(ws, input_is_latent = True)
def gen_slerp_w(G, device, nsteps = 5, seeds = [0, 2], styledim = 512, truncation = 1.0, trunc_mean = 4096):
  with torch.no_grad():
    G.eval()
    torch.manual_seed(seeds[0])
    start = torch.randn(1, styledim, device = device)
    torch.manual_seed(seeds[1])
    end = torch.randn(1, styledim, device = device)

    # pass through style network
    start_w = G.style(start)
    end_w = G.style(end)

    ws = spherical_interp(steps = nsteps, start = start_w.cpu(), end = end_w.cpu())
    ws = torch.stack(ws)
    ws = ws.to(torch.device('cuda'))
    for i in tqdm(range(nsteps)):

      sample, _ = G(
          [ws[i]],
          truncation = truncation,
          truncation_latent = trunc_mean,
          input_is_latent = True
      )

      utils.save_image(
          sample,
          f"sample_spherical_w/{str(i).zfill(4)}.png",
          nrow = nsteps,
          normalize = True,
          range = (-1, 1),
      )

if not os.path.exists("/sg/sample_spherical/"):
  os.mkdir("/sg/sample_spherical/")
if not os.path.exists("/sg/sample_spherical_w/"):
  os.mkdir("/sg/sample_spherical_w/")

gen_slerp_z(G = G, device = "cuda", nsteps = 25)
gen_slerp_w(G = G, device = "cuda", nsteps = 25)

# generate gif
fp_in = "/sg/sample_spherical/*.png"
fp_out = "/sg/sample_spherical/slerp.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=200, loop=0)

# generate gif (should probably wrap this in function at this point)
fp_in = "/sg/sample_spherical_w/*.png"
fp_out = "/sg/sample_spherical_w/slerp_w.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=200, loop=0)

if not os.path.exists("/sg/zippedfiles"):
    os.mkdir("/sg/zippedfiles")

# function to combine multiple images
def imgcombine(path):
  fp_in = os.path.join(path, "*.png")
  fp_out = os.path.join(path, "combined.png")
  img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
  widths, heights = zip(*(i.size for i in imgs))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in imgs:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save(fp_out)

imgcombine("/sg/sample_spherical_w/")
imgcombine("/sg/sample_spherical/")
imgcombine("/sg/sample/")
imgcombine("/sg/sample_w/")

# function to combine the combined images
def imgcombine(path):
  fp_in = os.path.join(path, "*.png")
  fp_out = os.path.join(path, "combined.png")
  img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
  widths, heights = zip(*(i.size for i in imgs))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in imgs:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save(fp_out)