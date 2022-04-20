import cv2
import networkx as nx
import numpy as np

import MRF
from typing import List, Union
import copy
from torchvision import datasets
import matplotlib.pyplot as plt


class UnknownPixelNode(MRF.RandomNode):
    value: float

    def __init__(self, value):
        super().__init__()
        self.value = value


class KnownPixelNode(MRF.ObservedNode):
    value: float

    def __init__(self, value):
        super().__init__()
        self.value = value


class LatentPixelFactor(MRF.BinaryFactor):

    def __init__(self, gamma, beta, xn: UnknownPixelNode, xm: UnknownPixelNode):
        self.gamma = gamma
        self.beta = beta
        self.a = xn
        self.b = xm

    def evalueate(self):
        # TODO: Account for beta
        return self.gamma * (self.a.value - self.b.value) ** 2

    def condition(self):
        pass


class ImageConsistencyFactor(MRF.BinaryFactor):

    def __init__(self, sigma, xn: UnknownPixelNode, dn: KnownPixelNode):
        self.sigma = sigma
        self.a = xn
        self.b = dn

    def evalueate(self):
        return ((self.a.value - self.b.value) ** 2) / (2 * self.sigma ** 2)

    def condition(self):
        if self.a is None:
            raise ValueError("Can not condition on deterministic variable")
        if self.b is None:
            return


def compute_from_neighbourhood(observed_pixel: KnownPixelNode,
                               neighbouring_pixels: List[UnknownPixelNode],
                               initial_guess_pixel_to_be_estimated: UnknownPixelNode,
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
    # sum_of_neighbours = sum(map(lambda x: x.value, neighbouring_pixels))
    sum_of_neighbours = 0
    M = 0
    for neighbour in neighbouring_pixels:
        if (float(neighbour.value) - float(initial_guess_pixel_to_be_estimated.value)) ** 2 < smoothing_factor.beta:
            sum_of_neighbours += neighbour.value
            M += 1
        else:
            print(f"LS: {(float(neighbour.value) - float(initial_guess_pixel_to_be_estimated.value)) ** 2},\t beta:{smoothing_factor.beta}.")
    # TODO Account for beta

    res = (observed_pixel.value + 2 * smoothing_factor.gamma * (
            observed_image_factor.sigma ** 2) * sum_of_neighbours) / (
                  1 + 2 * smoothing_factor.gamma * M * (observed_image_factor.sigma ** 2))
    # print("res:\t", res, ", gamma:\t", smoothing_factor.gamma, ", sigma:\t", observed_image_factor.sigma,
    #       ", sum of neighbours:\t", sum_of_neighbours, ", M:\t", len(neighbouring_pixels), "pixel value:\t",
    #       observed_pixel.value)
    return res


# def get_neighbours(mrf: MRF.MRF, node: MRF.RandomNode):
#     neigbours = []
#     for factor in mrf.factors:
#         if isinstance(factor, ImageConsistencyFactor):
#             continue
#         if id(factor.xn) == id(node):
#             neigbours.append(factor.xm)
#         elif id(factor.xm) == id(node):
#             neigbours.append(factor.xn)
#     return neigbours
#
# def get_observation(mrf: MRF.MRF, node: MRF.RandomNode):
#     for factor in mrf.factors:
#         if isinstance(factor, LatentPixelFactor):
#             continue
#         else:
#             if id(factor.xn) == id(node):
#                 return factor.dn
#     raise ValueError("The node is not found in the MRF.")


def icm(mrf: MRF.MRF, shape):
    # Setup:
    gamma = 100
    beta = 2000
    sigma = 1
    smoothing_factor = LatentPixelFactor(gamma, beta, None, None)
    intensity_factor = ImageConsistencyFactor(sigma, None, None)

    iterations = 5
    for i in range(iterations):
        img_i = img_from_mrf(mrf, shape)
        img_i = img_i.astype(float)
        img_i = img_i / img_i.max()
        cv2.imshow("iter" + str(i), img_i)
        new_mrf = copy.deepcopy(mrf)
        # for random_node in [node for node in new_mrf.nodes if isinstance(node, MRF.RandomNode)]:
        for e in range(len(mrf.nodes)):
            if isinstance(mrf.nodes[e], MRF.RandomNode):
                # neighbourhood = get_neighbours(mrf, mrf.nodes[e])
                neighbourhood = []
                observation_node = None
                neighbours = list(nx.all_neighbors(mrf.graph, mrf.nodes[e]))
                for neighbour in neighbours:
                    if isinstance(neighbour, UnknownPixelNode):
                        neighbourhood.append(neighbour)
                    elif isinstance(neighbour, KnownPixelNode):
                        observation_node = neighbour
                    else:
                        raise NotImplementedError
                # observation_node = get_observation(mrf, mrf.nodes[e])
                new_mrf.nodes[e].value = compute_from_neighbourhood(observation_node, neighbourhood, mrf.nodes[e],
                                                                    smoothing_factor,
                                                                    intensity_factor)
        mrf = new_mrf
    return mrf


def mrf_from_img(img: np.ndarray, beta, gamma, sigma) -> MRF.MRF:
    nodes = []
    factors = []
    node_grid = []
    mrf = MRF.MRF()
    for x0 in range(img.shape[0]):
        node_grid.append([])
        for x1 in range(img.shape[1]):
            new_random_node = UnknownPixelNode(img[x0, x1])
            node_grid[x0].append(new_random_node)
            new_deterministic_node = KnownPixelNode(img[x0, x1])
            mrf.add_node(new_random_node)
            mrf.add_node(new_deterministic_node)
            # nodes.append(new_random_node)
            # nodes.append(new_deterministic_node)

            new_intensity_factor = ImageConsistencyFactor(sigma, new_random_node, new_deterministic_node)
            # factors.append(new_intensity_factor)
            mrf.add_factor(new_intensity_factor)

            if x0 > 0:
                new_left_factor = LatentPixelFactor(gamma, beta, new_random_node, node_grid[x0 - 1][x1])
                mrf.add_factor(new_left_factor)
                # factors.append(new_left_factor)
            if x1 > 0:
                new_up_factor = LatentPixelFactor(gamma, beta, new_random_node, node_grid[x0][x1 - 1])
                # factors.append(new_up_factor)
                mrf.add_factor(new_up_factor)

    # subax1 = plt.subplot(121)
    # nx.draw(mrf.graph,node_color=[0 if isinstance(node, KnownPixelNode) else 1 for node in mrf.graph.nodes])
    # plt.show()

    return mrf


def img_from_mrf(mrf, shape) -> np.ndarray:
    img = np.array([node.value for node in mrf.nodes if isinstance(node, MRF.RandomNode)]).reshape(shape)
    return img


def main():
    dataset = datasets.Country211(
        root="data",
        # annFile=False
        # train=True,
        download=True,
    )
    pic = dataset[0][0]
    img = np.array(pic)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.pyrDown(img)
    shape = img.shape

    cv2.imshow("initial", img)

    mrf = mrf_from_img(img, 1, 1, 1)

    parital_img = img_from_mrf(mrf, shape)
    cv2.imshow("partial", parital_img)

    new_mrf = icm(mrf, shape)

    new_img = img_from_mrf(new_mrf, shape)
    new_img = new_img.astype(float)
    new_img = new_img / new_img.max()
    cv2.imshow("final", new_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
