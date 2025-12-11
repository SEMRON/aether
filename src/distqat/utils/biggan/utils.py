from torchvision import transforms

class CenterCropLongEdge(object):
  def __call__(self, img):
    return transforms.functional.center_crop(img, min(img.size))
