# Learning maximally monotone operators for image recovery

## Summary
We train a denoiser while imposing a firm-nonexpansiveness penalization through a Jacobian regularization term.
Given the denoiser `J`, we impose a 1-Lipschitz regularization on `Q = 2J-I`, where `I` is the identity.
The Lipschitz constant of `Q` is estimated via the spectral norm of its Jacobian.

The resulting denoiser is then plugged in a PnP forward-backward algorithm.

**Link to paper: [https://arxiv.org/pdf/2012.13247.pdf](https://arxiv.org/pdf/2012.13247.pdf)**

## Usage

#### Testing
In order to test the PnP-FB algorithm, in the grayscale case run:
```bash
python3 test_pnp.py --architecture='DnCNN_nobn' --n_ch=1 --noise_level=0.01 --noise_level_den=0.009
```
or in the color case run:
```bash
python3 test_pnp.py --architecture='DnCNN_nobn' --n_ch=3 --noise_level=0.01 --noise_level_den=0.007
```

#### Training
In order to train a DnCNN (without BN) as in the paper, for denoising grayscale images with noise level 0.01 and $\lambda=1e-5$ and $\epsilon=-0.05$, run:
```bash
python3 train.py --architecture='DnCNN_nobn' --n_ch=1  --epsilon=-0.05 --lambdajr=1e-5  --noise_level=0.01
```
Note that in the paper, we first pretrained the denoisers with `--lambdajr=0`, which improved results, and that we used a batch of size 100.

## Setup

This code relies on
```bash
- numpy
- torch >= 1.1
- torchvision > 0.3
- imageio
- PIL
- scipy
- scikit-image
```

In order to use this code, you need to update the paths in the [configfiles/setup.json](https://github.com/basp-group/PnP-MMO-imaging/blob/main/configfiles/setup.json) config file:
```json
{
"path_dataset": "/pth/to/your/ImageNet/val/"
, "path_dataset_pnp": "path/to/your/test/dataset/"
, "pattern_red": "*.jpg"
, "root_folder": "path/to/your/local/project/"
}
```

The current code assumes that:
- the weights are in a folder `checkpoints/pretrained/` at the root of your project (see the loading of the networks in the [Denoiser](https://github.com/basp-group/PnP-MMO-imaging/blob/main/optim/tools.py#L77) class);
- the blur kernels are in a folder `blur_models/` at the root of your project.

## Datasets & models
You can download the models and blur kernels [here](https://drive.google.com/drive/folders/1jNdx8NjqYueptsfjt5X_A_WJGUacJYX5?usp=sharing).

Models were trained on the ImageNet val set, you can download it there: [http://www.image-net.org/challenges/LSVRC/2012/](http://www.image-net.org/challenges/LSVRC/2012/)

The BSD300 dataset can be found there: [https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz)

## Citation & contact

```bibtex
@article{pesquet2020learning,
  title={Learning Maximally Monotone Operators for Image Recovery},
  author={Pesquet, Jean-Christophe and Repetti, Audrey and Terris, Matthieu and Wiaux, Yves},
  journal={to appear in SIAM J. on Imaging Sci.},
  year={2021}
}
```

Don't hesitate to contact me (mt114@hw.ac.uk)<sup>[*](#footnote)</sup> if you have any question!

**License:** GNU General Public License v3.0.

<a name="footnote"><sup>*</sup></a> This address might be deprecated soon; in this case queries can be addressed to y.wiaux@hw.ac.uk.