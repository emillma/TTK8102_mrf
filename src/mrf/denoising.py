import cv2
import numpy as np

import MRF
from typing import List
import copy
from torchvision import datasets

class UnknownPixelNode(MRF.RandomNode):
    value: float

    def __init__(self, value):
        self.value = value


class KnownPixelNode(MRF.ObservedNode):
    value: float

    def __init__(self, value):
        self.value = value

class LatentPixelFactor(MRF.Factor):

    def __init__(self, gamma, beta, xn: UnknownPixelNode, xm: UnknownPixelNode):
        self.gamma = gamma
        self.beta = beta
        self.xn = xn
        self.xm = xm

    def evalueate(self):
        # TODO: Account for beta
        return self.gamma * (self.xm.value - self.xn.value) ** 2

    def condition(self, A: UnknownPixelNode, B: UnknownPixelNode):
        pass


class ImageConsistencyFactor(MRF.Factor):

    def __init__(self, sigma, xn: UnknownPixelNode, dn: KnownPixelNode):
        self.sigma = sigma
        self.xn = xn
        self.dn = dn

    def evalueate(self):
        return ((self.xn.value - self.dn.value) ** 2) / (2 * self.sigma ** 2)

    def condition(self, A: UnknownPixelNode, B: KnownPixelNode):
        if B is None:
            raise ValueError("Can not condition on deterministic variable")
        if A is None:
            return


def compute_from_neighbourhood(observed_pixel: KnownPixelNode,
                               neighbouring_pixels: List[UnknownPixelNode],
                               smoothing_factor: LatentPixelFactor,
                               observed_image_factor: ImageConsistencyFactor):
    """
    This is created from the formulas in https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV0809/ORCHARD/
    :param observed_pixel:
    :param neighbouring_pixels:
    :param smoothing_factor:
    :param observed_image_factor:
    :return:
    """
    sum_of_neighbours = sum(map(lambda x: x.value, neighbouring_pixels))
    # TODO Account for beta

    return (2 * smoothing_factor.gamma * (
            observed_image_factor.sigma ** 2) * sum_of_neighbours + observed_pixel.value) / (
                   1 - 2 * smoothing_factor.gamma * len(neighbouring_pixels) * (observed_image_factor.sigma ** 2))

def get_neighbours(mrf: MRF.MRF, node: MRF.RandomNode):
    neigbours = []
    for factor in mrf.factors:
        if isinstance(factor, ImageConsistencyFactor):
            continue
        if id(factor.xn) == id(node):
            neigbours.append(factor.xm)
        elif id(factor.xm) == id(node):
            neigbours.append(factor.xn)
    return neigbours

def get_observation(mrf: MRF.MRF, node: MRF.RandomNode):
    for factor in mrf.factors:
        if isinstance(factor, LatentPixelFactor):
            continue
        else:
            if id(factor.xn) == id(node):
                return factor.dn
    raise ValueError("The node is not found in the MRF.")




def icm(mrf: MRF.MRF):
    # Setup:
    gamma = 1
    beta = 1000
    sigma = 1
    smoothing_factor = LatentPixelFactor(gamma, beta, None, None)
    intensity_factor = ImageConsistencyFactor(sigma, None, None)

    iterations = 10
    for i in range(iterations):
        new_mrf = copy.deepcopy(mrf)
        for random_node in [node for node in new_mrf.nodes if isinstance(node, MRF.RandomNode)]:
            neighbourhood = get_neighbours(new_mrf, random_node)
            observation_node = get_observation(new_mrf, random_node)
            random_node.value = compute_from_neighbourhood(observation_node, neighbourhood, smoothing_factor, intensity_factor)
        mrf  = new_mrf # TODO: This is not working, we need to refer to the old mrf when calculating the new value
    return mrf

def mrf_from_img(img: np.ndarray, beta, gamma, sigma) -> MRF.MRF:
    nodes = []
    factors = []
    node_grid = []
    for x0 in range(img.shape[0]):
        node_grid.append([])
        for x1 in range(img.shape[1]):
            new_random_node = UnknownPixelNode(img[x0, x1])
            node_grid[x0].append(new_random_node)
            new_deterministic_node = KnownPixelNode(img[x0, x1])
            nodes.append(new_random_node)
            nodes.append(new_deterministic_node)

            new_intensity_factor = ImageConsistencyFactor(sigma, new_random_node, new_deterministic_node)
            factors.append(new_intensity_factor)

            if x0 > 0:
                new_left_factor = LatentPixelFactor(gamma, beta, new_random_node, node_grid[x0][x1-1])
                factors.append(new_left_factor)
            if x1 > 0:
                new_up_factor = LatentPixelFactor(gamma,beta,new_random_node, node_grid[x0-1][x1])
                factors.append(new_up_factor)

    return MRF.MRF(nodes, factors)

def img_from_mrf(mrf, shape):
    img = np.array([node.value for node in mrf.nodes if isinstance(node, MRF.RandomNode)]).reshape(shape)
    return img


def main():
    dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
    )
    pic = dataset[0][0]
    img = np.array(pic)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = img.shape

    cv2.imshow("initial", img)

    mrf = mrf_from_img(img, 1, 1, 1)

    new_mrf = icm(mrf)

    new_img = img_from_mrf(new_mrf, shape)
    cv2.imshow("final", new_img)
    cv2.waitKey(0)




if __name__ == '__main__':
    main()