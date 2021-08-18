import numpy as np
import scipy
import scipy.io


def forward_conv_circ(xo, a):
    """
    Implements the linear operator A:xo -> xo*a, where * is the circular convolution
    """
    d = len(np.shape(xo))
    l = a.shape[0]
    if d == 2:
        xo = np.pad(xo, ((l//2+1, l//2), (l//2+1, l//2)), 'wrap')
    else:
        xo = np.pad(xo, ((0, 0), (l//2+1, l//2), (l//2+1, l//2)), 'wrap')
    A = np.fft.fft2(a, [xo.shape[-2], xo.shape[-1]])
    if d == 2:
        y = np.real(np.fft.ifft2(A * np.fft.fft2(xo)))
    else:
        y = np.zeros([xo.shape[0], xo.shape[1], xo.shape[2]])
        for i in range(xo.shape[0]): 
            y[i, ...] = np.real(np.fft.ifft2(A * np.fft.fft2(xo[i, ...])))
    return y[..., l:, l:]


def backward_conv_circ(xo, a):
    """
    Implements the adjoint A^* of the linear operator A:xo -> xo*a, where * is the circular convolution
    """
    d = len(np.shape(xo))
    l = a.shape[0]
    if d == 2:
        xo = np.pad(xo, ((l//2, l//2), (l//2, l//2)), 'wrap')
    else:
        xo = np.pad(xo, ((0, 0), (l//2, l//2), (l//2, l//2)), 'wrap')
    A = np.fft.fft2(a, [xo.shape[-2], xo.shape[-1]])
    if d == 2:
        y = np.real(np.fft.ifft2(np.conj(A) * np.fft.fft2(xo)))
    else:
        y = np.zeros([xo.shape[0], xo.shape[1], xo.shape[2]])
        for i in range(xo.shape[0]):
            y[i, ...] = np.real(np.fft.ifft2(np.conj(A) * np.fft.fft2(xo[i, ...])))
    return y[..., :-l+1, :-l+1]


def get_operators(type_op='circular_deconvolution', sigma=None, pth_kernel='blur_models/blur_1.mat', shape=(256, 256), cross_cor=True):
    """
    Returns the forward measurement operator, the backward (adjoint) operator as well as the noise.
    Feel free to add your new operator class!
    """

    if 'circular_deconvolution' in type_op:
        h = scipy.io.loadmat(pth_kernel)
        h = np.array(h['blur'])

        if cross_cor:  # This is when the forward model contains a cross correlation and not a convolution (case of Corbineau's paper)
            h = np.flip(h, 1)
            h = np.flip(h, 0).copy()  # Contiguous

        def forward_op(x): return forward_conv_circ(x, h)
        def backward_op(x): return backward_conv_circ(x, h)
        n = sigma*np.random.randn(*shape) if sigma is not None else None

        return forward_op, backward_op, n
    else:
        raise ValueError('Unknown operator type!')
