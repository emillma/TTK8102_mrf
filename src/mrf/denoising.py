import cv2
import networkx as nx
import numpy as np

import MRF
from typing import List, Union
import copy
from torchvision import datasets
import matplotlib.pyplot as plt
from dataclasses import dataclass


# TODO: Somehow make this a dataclass
# @dataclass(frozen=True)
class UnknownPixelNode(MRF.RandomNode):
    value: float

    def __init__(self, value):
        super().__init__()
        self.value = value


# TODO: Somehow make this a dataclass
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


def optimal_pixel_value(node_to_estimate: UnknownPixelNode,
                        smoothing_factors: List[LatentPixelFactor],
                        observed_image_factor: ImageConsistencyFactor):
    sum_of_neighbours = 0
    M = 0
    for neighbour_factor in smoothing_factors:
        if (float(neighbour_factor.get_other_node(node_to_estimate).value) - float(
                node_to_estimate.value)) ** 2 < neighbour_factor.beta:
            sum_of_neighbours += neighbour_factor.get_other_node(node_to_estimate).value
            M += 1
    print(f"{float(neighbour_factor.get_other_node(node_to_estimate).value)}, {float(node_to_estimate.value)}")
    # We assume that gamma is the same for all the neighbourfactors
    res = (observed_image_factor.get_other_node(node_to_estimate).value + 2 * smoothing_factors[0].gamma * (
            observed_image_factor.sigma ** 2) * sum_of_neighbours) / (
                  1 + 2 * smoothing_factors[0].gamma * M * (observed_image_factor.sigma ** 2))
    # print("res:\t", res, ", gamma:\t", smoothing_factor.gamma, ", sigma:\t", observed_image_factor.sigma,
    #       ", sum of neighbours:\t", sum_of_neighbours, ", M:\t", len(neighbouring_pixels), "pixel value:\t",
    #       observed_pixel.value)
    return res


def icm(mrf: MRF.MRF, shape):
    iterations = 20
    for i in range(iterations):
        img_i = img_from_mrf(mrf, shape)
        img_i = img_i.astype(float)
        img_i = img_i / img_i.max()
        cv2.imshow("iter" + str(i), img_i)
        new_mrf = copy.deepcopy(mrf)
        for e in range(len(mrf.nodes)):
            if isinstance(mrf.nodes[e], MRF.RandomNode):

                latentPixelFactors = []
                observedPixelFactor = None
                adjecent_factors = mrf.graph.adj[mrf.nodes[e]]
                for neighbour in adjecent_factors:
                    if isinstance(neighbour, UnknownPixelNode):
                        latentPixelFactors.append(adjecent_factors[neighbour]['factor'])
                    elif isinstance(neighbour, KnownPixelNode):
                        observedPixelFactor = adjecent_factors[neighbour]['factor']
                    else:
                        raise NotImplementedError
                new_mrf.nodes[e].value = optimal_pixel_value(mrf.nodes[e],
                                                             latentPixelFactors,
                                                             observedPixelFactor)
        mrf = new_mrf
    return mrf


def mrf_from_img(img: np.ndarray, beta, gamma, sigma, initial_value) -> MRF.MRF:
    node_grid = []
    mrf = MRF.MRF()
    for x0 in range(img.shape[0]):
        node_grid.append([])
        for x1 in range(img.shape[1]):
            new_random_node = UnknownPixelNode(img[x0, x1] if initial_value == 0 else initial_value)
            node_grid[x0].append(new_random_node)
            new_deterministic_node = KnownPixelNode(img[x0, x1])
            mrf.add_node(new_random_node)
            mrf.add_node(new_deterministic_node)

            new_intensity_factor = ImageConsistencyFactor(sigma, new_random_node, new_deterministic_node)
            mrf.add_factor(new_intensity_factor)

            if x0 > 0:
                new_left_factor = LatentPixelFactor(gamma, beta, new_random_node, node_grid[x0 - 1][x1])
                mrf.add_factor(new_left_factor)
            if x1 > 0:
                new_up_factor = LatentPixelFactor(gamma, beta, new_random_node, node_grid[x0][x1 - 1])
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
        download=True,
    )
    # dataset = datasets.CIFAR10(
    #     root="data",
    #     train=True,
    #     download=True,
    # )
    pic = dataset[0][0]
    img = np.array(pic)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.pyrDown(img)
    shape = img.shape

    cv2.imshow("initial", img)
    gamma = 100
    beta = 2000
    sigma = 1

    mrf = mrf_from_img(img, beta, gamma, sigma, 0)

    new_mrf = icm(mrf, shape)

    new_img = img_from_mrf(new_mrf, shape)
    new_img = new_img.astype(float)
    new_img = new_img / new_img.max()
    cv2.imshow("final", new_img)
    cv2.waitKey(0)

def main2():
    dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
    )
    pic = dataset[0][0]
    img = np.array(pic)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)

    cv2.imshow("Example_img", img)

    gamma = 10
    beta = 2000
    sigma = 1
    mrf = mrf_from_img(img, beta, gamma, sigma, 0)

    plt.figure(figsize=(10,10,))
    subax1 = plt.subplot(121)
    subax1.set_aspect(1)
    pos_dict = dict([(node, (4* ((i / 2) % img.shape[0]), 4* (((i / 2) // img.shape[0]) + int(isinstance(node, KnownPixelNode)) * 0.5))) for (i, node,) in enumerate(mrf.nodes)])
    color_array = [0 if isinstance(node, KnownPixelNode) else 1 for node in mrf.graph.nodes]
    nx.draw(mrf.graph,pos_dict, node_color=color_array, node_size=50)
    plt.show()


if __name__ == '__main__':
    main()
