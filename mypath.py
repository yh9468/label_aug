class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/dataset/VOC0712/VOC2012'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/dataset/SBD/'  # folder that contains dataset/
        elif dataset == 'cityscapes':
            return '/root/volume/Cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
