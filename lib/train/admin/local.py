
class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = './results/'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = '/home/bit231/tracking_dataset/LaSOT'
        self.got10k_dir = '/home/bit231/tracking_dataset/GOT-10k/full_data/train_data'
        self.trackingnet_dir = '/home/bit231/tracking_dataset/TrackingNet'
        self.coco_dir = '/home/bit231/tracking_dataset/coco'
  

                

