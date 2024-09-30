# This file is autogenerated by the command `make fix-copies`, do not edit.
from ..utils import requires_backends


class LayoutLMv2Model:

  def __init__(self, *args, **kwargs):
    requires_backends(self, ["detectron2"])

  @classmethod
  def from_pretrained(cls, *args, **kwargs):
    requires_backends(cls, ["detectron2"])
