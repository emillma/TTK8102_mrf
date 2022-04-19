import MRF
from typing import List


class UnknownPixelNode(MRF.RandomNode):
    value: float


class KnownPixelNode(MRF.ObservedNode):
    value: float


class LatentPixelFactor(MRF.Factor):

    def __init__(self, gamma, beta):
        self.gamma = gamma
        self.beta = beta

    def evalueate(self, xn: UnknownPixelNode, xm: UnknownPixelNode):
        # TODO: Account for beta
        return self.gamma * (xm.value - xn.value) ** 2

    def condition(self, A: UnknownPixelNode, B: UnknownPixelNode):
        pass


class ImageConsistencyFactor(MRF.Factor):

    def __init__(self, sigma):
        self.sigma = sigma

    def evalueate(self, xn: UnknownPixelNode, dn: KnownPixelNode):
        return ((xn.value - dn.value) ** 2) / (2 * self.sigma ** 2)

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



