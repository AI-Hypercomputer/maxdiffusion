# This file is autogenerated by the command `make fix-copies`, do not edit.
from ..utils import DummyObject, requires_backends


class Pop2PianoFeatureExtractor(metaclass=DummyObject):
  _backends = ["essentia", "librosa", "pretty_midi", "scipy", "torch"]

  def __init__(self, *args, **kwargs):
    requires_backends(self, ["essentia", "librosa", "pretty_midi", "scipy", "torch"])


class Pop2PianoTokenizer(metaclass=DummyObject):
  _backends = ["essentia", "librosa", "pretty_midi", "scipy", "torch"]

  def __init__(self, *args, **kwargs):
    requires_backends(self, ["essentia", "librosa", "pretty_midi", "scipy", "torch"])


class Pop2PianoProcessor(metaclass=DummyObject):
  _backends = ["essentia", "librosa", "pretty_midi", "scipy", "torch"]

  def __init__(self, *args, **kwargs):
    requires_backends(self, ["essentia", "librosa", "pretty_midi", "scipy", "torch"])
