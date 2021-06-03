from models import small_convnet, average_convnet


class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(name, input_shape, num_classes, max_objects):
        if name == 'small_convnet':
            return small_convnet(input_shape=input_shape,
                                 num_classes=num_classes,
                                 max_objects=max_objects)
        elif name == 'average_convnet':
            return average_convnet(input_shape=input_shape,
                                   num_classes=num_classes,
                                   max_objects=max_objects)
