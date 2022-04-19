import MRF


class UnknownPixelNode(MRF.Node):
    value: float


class KnownPixelNode(MRF.Node):
    value: float


class LatentPixelFactor(MRF.Factor):
    def evalueate(self, A: UnknownPixelNode, B: UnknownPixelNode):
        return (A.value - B.value) ** 2

    def condition(self, A: UnknownPixelNode, B: UnknownPixelNode):
        pass

class ImageConsistencyFactor(MRF.Factor):
    def evalueate(self, A: UnknownPixelNode, B: KnownPixelNode):
        return (A.value - B.value) ** 2

    def condition(self, A: UnknownPixelNode, B: KnownPixelNode):
        pass
