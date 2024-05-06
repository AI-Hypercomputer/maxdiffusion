import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import os
import scipy
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

def concat_all_arrays(arrays: list[list[np.memmap, int]]):
    non_padded_array=[]
    for array in arrays:
        if type(array) !='list':
            return array
        padded_array = array[0]
        len = array[1]
        non_padded_array.append(padded_array[:len, :])
    return jnp.concat(non_padded_array, axis=0)

def compute_statistics_with_mmap(path, mmap_filname, params, apply_fn, batch_size=1, img_size=None):
    # need to read as byte for np.load
    if path.endswith(".npz"):
        with tf.io.gfile.GFile(path, 'rb') as f:
            stats = np.load(f)
        mu, sigma = stats["mu"], stats["sigma"]
        return mu, sigma
    
    preprocessing_fn = lambda x: x.astype(float) / 255
    image_data_generator = ImageDataGenerator(preprocessing_function=preprocessing_fn)
    directory_iterator = image_data_generator.flow_from_directory(
        path, batch_size=batch_size, target_size=img_size, shuffle=False, interpolation="bilinear"
    )
    assert directory_iterator.samples > 0, "No images found. Make sure your images are within a subdirectory."

    get_batch_fn = lambda: directory_iterator.next()[0]
    num_activations = directory_iterator.samples
    num_batches = len(directory_iterator)
    dtype = 'float32'
    activation_dim = 2048

    mm = np.memmap(mmap_filname, dtype=dtype, mode='w+', shape=(num_activations, activation_dim))

    activation_sum = np.zeros((activation_dim))
    for i in tqdm(range(num_batches)):
        x = get_batch_fn()
        x = np.asarray(x)
        x = 2 * x - 1
        activation_batch = apply_fn(params, jax.lax.stop_gradient(x))
        activation_batch = activation_batch.squeeze(axis=1).squeeze(axis=1)

        current_batch_size = activation_batch.shape[0]
        start_index = i * batch_size
        end_index = start_index + current_batch_size
        mm[start_index : end_index] = activation_batch

        activation_sum += activation_batch.sum(axis=0)

    sigma = np.cov(mm, rowvar=False)
    sigma = sigma * num_activations
    sigma = jax.experimental.multihost_utils.process_allgather(sigma)
    sigma = jnp.sum(sigma, axis=0)

    num_activations = jax.experimental.multihost_utils.process_allgather(num_activations)
    num_activations = jnp.sum(num_activations)

    activation_sum = jax.experimental.multihost_utils.process_allgather(activation_sum)
    activation_sum = jnp.sum(activation_sum, axis=0)

    print(num_activations)
    
    mu = activation_sum / num_activations
    sigma = sigma / num_activations
    return mu, sigma

def compute_statistics(path, params, apply_fn, batch_size=1, img_size=None):
    if path.endswith(".npz"):
        # need to read as byte for np.load
        with tf.io.gfile.GFile(path, 'rb') as f:
            stats = np.load(f)
        mu, sigma = stats["mu"], stats["sigma"]
        return mu, sigma

    images = []
    for f in tqdm(os.listdir(path)):
        img = Image.open(os.path.join(path, f))
        # convert if only a single channel
        if img.mode == "L":
            img = img.convert("RGB")
        # resize if not the right size
        if img_size is not None and img.size[:2] != img_size:
            img = img.resize(
                size=(img_size[0], img_size[1]),
                resample=Image.Resampling.BILINEAR,
            )
        img = np.array(img) / 255.0
        images.append(img)

    num_batches = int(len(images) // batch_size)
    act = []
    for i in tqdm(range(num_batches)):
        x = images[i * batch_size : i * batch_size + batch_size]
        x = np.asarray(x)
        x = 2 * x - 1
        pred = apply_fn(params, jax.lax.stop_gradient(x))
        act.append(pred.squeeze(1).squeeze(1))
    act = jnp.concatenate(act, axis=0)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6):
    # Taken from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_1d(sigma1)
    sigma2 = np.atleast_1d(sigma2)

    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2
    

    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
    covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean