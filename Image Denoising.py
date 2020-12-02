import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import pickle
from skimage.util import view_as_windows as viewW
import time


def images_example(path='train_images.pickle'):
    """
    A function demonstrating how to access to image data supplied in this exercise.
    :param path: The path to the pickle file.
    """
    patch_size = (8, 8)

    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    patches = sample_patches(train_pictures, psize=patch_size, n=20000)

    plt.figure()
    plt.imshow(train_pictures[0])
    plt.title("Picture Example")

    plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(patches[:,i].reshape(patch_size), cmap='gray')
        plt.title("Patch Example")
    plt.show()


def im2col(A, window, stepsize=1):
    """
    an im2col function, transferring an image to patches of size window (length
    2 list). the step size is the stride of the sliding window.
    :param A: The original image (NxM size matrix of pixel values).
    :param window: Length 2 list of 2D window size.
    :param stepsize: The step size for choosing patches (default is 1).
    :return: A (heightXwidth)x(NxM) matrix of image patches.
    """
    return viewW(np.ascontiguousarray(A), (window[0], window[1])).reshape(-1,
                        window[0] * window[1]).T[:, ::stepsize]


def grayscale_and_standardize(images, remove_mean=True):
    """
    The function receives a list of RGB images and returns the images after
    grayscale, centering (mean 0) and scaling (between -0.5 and 0.5).
    :param images: A list of images before standardisation.
    :param remove_mean: Whether or not to remove the mean (default is True).
    :return: A list of images after standardisation.
    """
    standard_images = []

    for image in images:
        standard_images.append((0.299 * image[:, :, 0] +
                                0.587 * image[:, :, 1] +
                                0.114 * image[:, :, 2]) / 255)

    sum = 0
    pixels = 0
    for image in standard_images:
        sum += np.sum(image)
        pixels += image.shape[0] * image.shape[1]
    dataset_mean_pixel = float(sum) / pixels

    if remove_mean:
        for image in standard_images:
            image -= np.matlib.repmat([dataset_mean_pixel], image.shape[0],
                                      image.shape[1])

    return standard_images


def sample_patches(images, psize=(8, 8), n=10000, remove_mean=True):
    """
    sample N p-sized patches from images after standardising them.

    :param images: a list of pictures (not standardised).
    :param psize: a tuple containing the size of the patches (default is 8x8).
    :param n: number of patches (default is 10000).
    :param remove_mean: whether the mean should be removed (default is True).
    :return: A matrix of n patches from the given images.
    """
    d = psize[0] * psize[1]
    patches = np.zeros((d, n))
    standardized = grayscale_and_standardize(images, remove_mean)

    shapes = []
    for pic in standardized:
        shapes.append(pic.shape)

    rand_pic_num = np.random.randint(0, len(standardized), n)
    rand_x = np.random.rand(n)
    rand_y = np.random.rand(n)

    for i in range(n):
        pic_id = rand_pic_num[i]
        pic_shape = shapes[pic_id]
        x = int(np.ceil(rand_x[i] * (pic_shape[0] - psize[1])))
        y = int(np.ceil(rand_y[i] * (pic_shape[1] - psize[0])))
        patches[:, i] = np.reshape(np.ascontiguousarray(
            standardized[pic_id][x:x + psize[0], y:y + psize[1]]), d)

    return patches


def denoise_image(Y, model, denoise_function, noise_std, patch_size=(8, 8)):
    """
    A function for denoising an image. The function accepts a noisy gray scale
    image, denoises the different patches of it and then reconstructs the image.

    :param Y: the noisy image.
    :param model: a Model object (MVN/ICA/GSM).
    :param denoise_function: a pointer to one of the denoising functions (that corresponds to the model).
    :param noise_std: the noise standard deviation parameter.
    :param patch_size: the size of the patch that the model was trained on (default is 8x8).
    :return: the denoised image, after each patch was denoised. Note, the denoised image is a bit
    smaller than the original one, since we lose the edges when we look at all of the patches
    (this happens during the im2col function).
    """
    (h, w) = np.shape(Y)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))

    # split the image into columns and denoise the columns:
    noisy_patches = im2col(Y, patch_size)
    denoised_patches = denoise_function(noisy_patches, model, noise_std)

    # reshape the denoised columns into a picture:
    x_hat = np.reshape(denoised_patches[middle_linear_index, :],
                       [cropped_h, cropped_w])

    return x_hat


def crop_image(X, patch_size=(8, 8)):
    """
    crop the original image to fit the size of the denoised image.
    :param X: The original picture.
    :param patch_size: The patch size used in the model, to know how much we need to crop.
    :return: The cropped image.
    """
    (h, w) = np.shape(X)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))
    columns = im2col(X, patch_size)
    return np.reshape(columns[middle_linear_index, :], [cropped_h, cropped_w])


def normalize_log_likelihoods(X):
    """
    Given a matrix in log space, return the matrix with normalized columns in
    log space.
    :param X: Matrix in log space to be normalised.
    :return: The matrix after normalization.
    """
    h, w = np.shape(X)
    return X - np.matlib.repmat(logsumexp(X, axis=0), h, 1)


def test_denoising(image, model, denoise_function, label,
                   noise_range=(0.01, 0.05, 0.1, 0.2), patch_size=(8, 8)):
    """
    A simple function for testing your denoising code. You can and should
    implement additional tests for your code.
    :param image: An image matrix.
    :param model: A trained model (MVN/ICA/GSM).
    :param denoise_function: The denoise function that corresponds to your model.
    :param noise_range: A tuple containing different noise parameters you wish
            to test your code on. default is (0.01, 0.05, 0.1, 0.2).
    :param patch_size: The size of the patches you've used in your model.
            Default is (8, 8).
    """
    h, w = np.shape(image)
    noisy_images = np.zeros((h, w, len(noise_range)))
    denoised_images = []
    cropped_original = crop_image(image, patch_size)
    n_mses = []
    d_mses = []

    # make the image noisy:
    for i in range(len(noise_range)):
        noisy_images[:, :, i] = image + (
        noise_range[i] * np.random.randn(h, w))

    # denoise the image:
    for i in range(len(noise_range)):
        denoised_images.append(
            denoise_image(noisy_images[:, :, i], model, denoise_function,
                          noise_range[i], patch_size))

    # calculate the MSE for each noise range:
    for i in range(len(noise_range)):
        print("noisy MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((crop_image(noisy_images[:, :, i],
                                  patch_size) - cropped_original) ** 2))
        n_mses.append(np.mean((crop_image(noisy_images[:, :, i],
                                  patch_size) - cropped_original) ** 2))
        print("denoised MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((cropped_original - denoised_images[i]) ** 2))
        d_mses.append(np.mean((cropped_original - denoised_images[i]) ** 2))

    plt.figure()
    for i in range(len(noise_range)):
        plt.subplot(2, len(noise_range), i + 1)
        plt.imshow(noisy_images[:, :, i], cmap='gray')
        plt.subplot(2, len(noise_range), i + 1 + len(noise_range))
        plt.imshow(denoised_images[i], cmap='gray')
    plt.title(label)
    plt.show()

    return noise_range, d_mses, n_mses


class MVN_Model:
    """
    A class that represents a Multivariate Gaussian Model, with all the parameters
    needed to specify the model.

    mean - a D sized vector with the mean of the gaussian.
    cov - a D-by-D matrix with the covariance matrix.
    """
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov


class GSM_Model:
    """
    A class that represents a GSM Model, with all the parameters needed to specify
    the model.

    cov - a k-by-D-by-D tensor with the k different covariance matrices. the
        covariance matrices should be scaled versions of each other.
    mix - k-length probability vector for the mixture of the gaussians.
    """
    def __init__(self, cov, mix):
        self.cov = cov
        self.mix = mix


class ICA_Model:
    """
    A class that represents an ICA Model, with all the parameters needed to specify
    the model.

    P - linear transformation of the sources. (X = P*S)
    vars - DxK matrix whose (d,k) element corresponds to the variance of the k'th
        gaussian in the d'th source.
    mix - DxK matrix whose (d,k) element corresponds to the weight of the k'th
        gaussian in d'th source.
    """
    def __init__(self, P, vars, mix):
        self.P = P
        self.vars = vars
        self.mix = mix


def MVN_log_likelihood(X, model):
    """
    Given image patches and a MVN model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A MVN_Model object.
    :return: The log likelihood of all the patches combined.
    """
    d = len(X)
    res_ll = 0
    log_det_sigma = np.log(np.linalg.det(model.cov))
    sigma_inverse = numpy.linalg.inv(model.cov)
    const = d * np.log(2 * np.pi)
    i = 0
    part_a = -float(X.shape[1] / 2)
    part_a = part_a * (const + log_det_sigma)
    for x in X.T:
        x_minus_u = x - model.mean
        ll_x = np.transpose(x_minus_u).dot(sigma_inverse)
        ll_x = ll_x.dot(x_minus_u)
        res_ll += ll_x
        i += 1

    sum = part_a + (-0.5 * res_ll)
    return sum


def GSM_log_likelihood(X, model):
    """
    Given image patches and a GSM model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A GSM_Model object.
    :return: The log likelihood of all the patches combined.
    """
    k = len(model.mix)
    d = len(X)
    pdf = np.zeros((X.shape[1], k))
    for i, pi_i in enumerate(model.mix):
        denominator = ((2 * np.pi) ** d) * np.linalg.det(model.cov[i]) ** 0.5
        inv_e = np.linalg.inv(model.cov[i])
        for j, x in enumerate(X.T):
            likelihood = np.exp((-0.5) * (x.dot(inv_e).dot(x.T))) / denominator
            pdf[j, i] = np.log(likelihood * pi_i)
            if np.isinf(pdf[j, i]):
                pdf[j, i] = 0

    weighted_sum = logsumexp(pdf, axis=1)
    ll = np.sum(weighted_sum)
    return ll
    # TODO: YOUR CODE HERE


def ICA_log_likelihood(X, model):
    """
    Given image patches and an ICA model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: An ICA_Model object.
    :return: The log likelihood of all the patches combined.
    """

    # TODO: YOUR CODE HERE


def calc_cov_mat(X, mean):
    e2 = np.cov(X)
    return e2


def pdf_mat(x, det_e, inv_e, denim_const):
    denominator = 0.5 * (np.log(denim_const) + np.log(det_e))
    if np.isinf(denominator) or np.isnan(denominator):
        z = 0
    tmp = x.dot(inv_e)
    p_x = np.zeros(len(tmp))
    for i in range(len(tmp)):
        p_x[i] = tmp[i].dot(x.T[:, i])
    p_x = (-0.5 * p_x) - denominator
    is_inf = np.isinf(p_x)
    is_nan = np.isnan(p_x)
    if is_inf.any() or is_nan.any():
        nans = np.where(is_nan)
        infs = np.where(is_inf)
        for i in nans:
            p_x[i] = -600
        for j in infs:
            p_x[j] = -600
    return p_x


def pdf_x(x, det_e, inv_e, denim_const):
    denominator = 0.5 * (np.log(denim_const) + np.log(det_e))
    if np.isinf(denominator) or np.isnan(denominator):
        z = 0
    tmp = x.dot(inv_e).dot(x.T)
    p_x = (-0.5 * tmp) - denominator
    if p_x == 0 or numpy.isinf(p_x) or np.isnan(p_x):
        p_x = np.random.randint(low=-600, high=-500)
    return p_x


def learn_MVN(X):
    """
    Learn a multivariate normal model, given a matrix of image patches.
    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :return: A trained MVN_Model object.
    """
    u = np.mean(X, axis=1)
    e = calc_cov_mat(X, u)
    model = MVN_Model(u, e)
    return model
    # TODO: YOUR CODE HERE


def m_step(X, c_i_y, cov):
    # c_y_sums = np.sum(c_i_y, axis=0)
    c_y_sums = logsumexp(c_i_y, axis=0)
    pi_y = c_y_sums - np.log(len(X))
    inv_e = np.linalg.inv(cov)
    x_inv = (X.T.dot(inv_e))
    # x_inv_x = np.dot(x_inv[, :], X[:, ])
    x_inv_x = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        x_inv_x[i] = x_inv[i, :].dot(X[:, i])
    # numerator = c_i_y.T.dot(x_inv_x)
    numerator = np.log(np.exp(c_i_y).T.dot(x_inv_x))
    # r_scales = numerator / (X.shape[0] * c_y_sums)
    r_scales = numerator - (np.log(X.shape[0]) + c_y_sums)
    return pi_y, r_scales


def e_step(X, pi_weights, cov, r_scales):
    denim_const = ((2 * np.pi) ** X.shape[0])
    c_i_y_mat = np.zeros((X.shape[1], len(pi_weights)))
    for i in range(len(pi_weights)):
        cur_cov = np.exp(r_scales[i]) * cov
        det_e = np.linalg.det(cur_cov)
        inv_e = np.linalg.inv(cur_cov)
        for j, x in enumerate(X.T):
            c_i_y_mat[j, i] = pdf_x(x, det_e, inv_e, denim_const)

        c_i_y_mat[:, i] += pi_weights[i]

    c_i_y_sums = logsumexp(c_i_y_mat, axis=1)
    c_i_y_mat = (c_i_y_mat.T - c_i_y_sums).T
    return c_i_y_mat


def learn_GSM(X, k):
    """
    Learn parameters for a Gaussian Scaling Mixture model for X using EM.

    GSM components share the variance, up to a scaling factor, so we only
    need to learn scaling factors and mixture proportions.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components of the GSM model.
    :return: A trained GSM_Model object.
    """
    e = calc_cov_mat(X, np.zeros(X.shape[0]))
    prev_pi = np.zeros(k)
    pi_weights = np.random.rand(k)
    pi_weights = np.log(pi_weights / np.sum(pi_weights)) # Probabilities sums to one
    r_scales = np.log(np.random.rand(k)) * 10
    prev_r = np.zeros(k)
    i = 1
    while not np.allclose(prev_pi, pi_weights) and not np.allclose(prev_r, r_scales) and i < 100:
        c_i_y = e_step(X, pi_weights, e, r_scales)
        prev_pi = pi_weights
        prev_r = r_scales
        pi_weights, r_scales = m_step(X, c_i_y, e)
        i += 1

    print("Total EM iterations executed: " + str(i))

    cov = np.zeros((len(pi_weights), len(X), len(X)))
    r_scales = np.exp(r_scales)
    for j in range(len(r_scales)):
        cov[j] = r_scales[j] * e

    model = GSM_Model(cov, pi_weights)
    print("finished learning")
    return model

    # TODO: YOUR CODE HERE


def learn_ICA(X, k):
    """
    Learn parameters for a complete invertible ICA model.

    We learn a matrix P such that X = P*S, where S are D independent sources
    And for each of the D coordinates we learn a mixture of K univariate
    0-mean gaussians using EM.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components in the source gaussian mixtures.
    :return: A trained ICA_Model object.
    """

    # TODO: YOUR CODE HERE


def MVN_Denoise(Y, mvn_model, noise_std):
    """
    Denoise every column in Y, assuming an MVN model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a single
    0-mean multi-variate normal distribution.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param mvn_model: The MVN_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """
    return calc_weiner(mvn_model.cov, noise_std, Y, mvn_model.mean, False)
    # TODO: YOUR CODE HERE


def calc_weiner(cov, std, Y, mean, is_gsm):
    res = np.zeros(Y.shape)
    sigma_inverse = numpy.linalg.inv(cov)
    variance = std ** 2
    i_mat = np.eye(len(sigma_inverse))
    weiner = np.linalg.inv(sigma_inverse + ((1 / variance) * i_mat))
    for i, y in enumerate(Y.T):
        denoised = weiner.dot(sigma_inverse.dot(mean) + ((1 / variance) * y))
        res[:, i] = denoised

    return res


def GSM_Denoise(Y, gsm_model, noise_std):
    """
    Denoise every column in Y, assuming a GSM model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a mixture of
    0-mean gaussian components sharing the same covariance up to a scaling factor.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param gsm_model: The GSM_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.

    """
    d = len(gsm_model.mix)
    denim_const = ((2 * np.pi) ** Y.shape[0])
    weiners = np.zeros((d, Y.shape[0], Y.shape[1]))
    for i in range(d):
        weiners[i] = calc_weiner(gsm_model.cov[i], noise_std, Y, np.zeros(len(Y)), True)

    c_i_res = np.zeros((Y.shape[1], d))
    for i in range(d):
        cur_cov = gsm_model.cov[i]
        eye = np.eye(cur_cov.shape[0]) * (noise_std ** 2)
        cur_cov += eye
        det_e = np.linalg.det(cur_cov)
        inv_e = np.linalg.inv(cur_cov)
        c_i_res[:, i] = pdf_mat(Y.T, det_e, inv_e, denim_const)
        c_i_res[:, i] += gsm_model.mix[i]

    c_i_sums = logsumexp(c_i_res, axis=1)
    c_i_res = (c_i_res.T - c_i_sums).T

    for k in range(d):
        weiners[k] = weiners[k] * np.exp(c_i_res[:, k])

    res = np.sum(weiners, axis=0)
    return res
    # TODO: YOUR CODE HERE


def ICA_Denoise(Y, ica_model, noise_std):
    """
    Denoise every column in Y, assuming an ICA model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by an ICA 0-mean
    mixture model.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param ica_model: The ICA_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """

    # TODO: YOUR CODE HERE


def train_GSM(psize):
    patch_size = (psize, psize)
    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)
    patches = sample_patches(train_pictures, psize=patch_size)
    trained = learn_GSM(patches, 5)
    ll = GSM_log_likelihood(patches, trained)
    print(ll)
    return trained


def train_MNV(psize):
    patch_size = (psize, psize)
    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)
    patches = sample_patches(train_pictures, psize=patch_size)
    trained = learn_MVN(patches)
    return trained


def compare_ll(gsm_model, mvn_model):
    patch_size = (8, 8)
    with open('test_images.pickle', 'rb') as f:
        test_pictures = pickle.load(f)
    patches = sample_patches(test_pictures, psize=patch_size)
    gsm_ll = GSM_log_likelihood(patches, gsm_model)
    mvn_ll = MVN_log_likelihood(patches, mvn_model)
    print(gsm_ll)
    print(mvn_ll)


def compare_mse(ranges, gsm_mse, mvn_mse, noise_mse):
    labels = ranges
    for i in range(len(noise_mse)):
        noise_mse[i] = min(gsm_mse[i] * 2, noise_mse[i])
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 3, gsm_mse, width=width / 3, label='GSM', align='center')
    rects2 = ax.bar(x, mvn_mse, width=width / 3, label='MVN', align='center')
    rects3 = ax.bar(x + width / 3, noise_mse, width=width / 3, label='noise', align='center')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('MSE')
    ax.set_title('MSE of denoised image by model')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    gsm = train_GSM(8)
    mvn = train_MNV(8)
    # compare_ll(gsm, mvn)
    with open('test_images.pickle', 'rb') as f:
        test_pictures = pickle.load(f)
    im_idx = np.random.randint(0, len(test_pictures))
    im = [test_pictures[im_idx]]
    im = grayscale_and_standardize(im)

    ranges, mvn_mse, noise_mse = test_denoising(im[0], mvn, MVN_Denoise, "MVN")
    _, gsm_mse, _ = test_denoising(im[0], gsm, GSM_Denoise, "GSM")

    compare_mse(ranges, gsm_mse, mvn_mse, noise_mse)

