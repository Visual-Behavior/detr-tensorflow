

def normalized_images(image, config):
    if config.normalized_method == "torch_resnet50":
        channel_avg = np.array([0.485, 0.456, 0.406])
        channel_std = np.array([0.229, 0.224, 0.225])
        image = (image / 255.0 - channel_avg) / channel_std
        return image
    elif config.normalized_method == "tf_resnet50":
        mean = [103.939, 116.779, 123.68]
        image = image[..., ::-1]
        image = image - mean
        return image
    else:
        raise Exception("Can't handler thid normalized method")