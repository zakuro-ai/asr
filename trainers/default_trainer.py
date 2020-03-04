import os

class DefaultTrainer:
    def __init__(self, model, args):
        self.args = args
        self.net = model.net
        self.main_proc = model._main_proc
        self.device = self.net.device
        # Instantiate variables for training
        self.distributed = args.world_size > 1
        self.save_folder = args.save_folder
        self.output_file = args.output_file
        self.manifest = args.manifest
        self.epochs = args.epochs
        self.train_manifest = args.train_manifest
        self.val_manifest = args.val_manifest
        self.model_path = args.model_path
        self.continue_from = args.continue_from if os.path.exists(args.continue_from) else None
        self.rank=args.rank
        self.gpu_rank=args.gpu_rank
        self.world_size=args.world_size
        # Instantiate default variables
        self.epoch_time = 0
        self.train_sampler = None
        self.train_loader = None
        self.test_loader = None
        self.end = None
        self.epoch_valid = False
        self.data = None

    def show(self):
        # print(self.net)
        print("================ VARS ===================")
        print('manifest:', self.manifest)
        print('distributed:', self.distributed)
        print('train_manifest:', self.train_manifest)
        print('val_manifest:', self.val_manifest)
        print('model_path:', self.model_path)
        print('continue_from:', self.continue_from)
        print('output_file:', self.output_file)
        print('main_proc:', self.main_proc)
        print('rank:', self.rank)
        print('gpu_rank:', self.gpu_rank)
        print('world_size:', self.world_size)
        print("==========================================")

    def __str__(self):
        print(self.net)

