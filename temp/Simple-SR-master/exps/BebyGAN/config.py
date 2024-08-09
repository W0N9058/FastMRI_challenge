from easydict import EasyDict as edict


class Config:
    # dataset
    DATASET = edict()
    DATASET.TYPE = 'MixDataset'
#     DATASET.DATASETS = ['DIV2K', 'Flickr2K']
    DATASET.DATASETS = ['FASTMRI']
#     DATASET.SPLITS = ['TRAIN', 'TRAIN']
    DATASET.SPLITS = ['TRAIN']
    DATASET.PHASE = 'train'
    DATASET.INPUT_HEIGHT = 384
    DATASET.INPUT_WIDTH = 384
    DATASET.SCALE = 1
    DATASET.REPEAT = 1
    DATASET.VALUE_RANGE = 0.001
    DATASET.SEED = 100

    # dataloader
    DATALOADER = edict()
    DATALOADER.IMG_PER_GPU = 1
    DATALOADER.NUM_WORKERS = 1

    # model
    MODEL = edict()
    MODEL.FLAT_KSIZE = 11
    MODEL.FLAT_STD = 0.025
    # generator
    MODEL.G = edict()
    MODEL.G.IN_CHANNEL = 1
    MODEL.G.OUT_CHANNEL = 1
    MODEL.G.N_CHANNEL = 8  # 64
    MODEL.G.N_BLOCK = 1  # 23
    MODEL.G.N_GROWTH_CHANNEL = 32
    
    # discriminator
    MODEL.D = edict()
    MODEL.D.IN_CHANNEL = 1
    MODEL.D.N_CHANNEL = 8  # 32
    MODEL.D.LOSS_TYPE = 'vanilla'  # vanilla | lsgan | wgan | wgan_softplus | hinge
    
    # best buddy loss, adversarial loss, back projection loss
    MODEL.BBL_WEIGHT = 1.0
    MODEL.BBL_ALPHA = 1.0
    MODEL.BBL_BETA = 1.0
    MODEL.BBL_KSIZE = 3
    MODEL.BBL_PAD = 0
    MODEL.BBL_STRIDE = 3
    MODEL.BBL_TYPE = 'l1'
    MODEL.ADV_LOSS_WEIGHT = 0.005
    MODEL.BACK_PROJECTION_LOSS_WEIGHT = 1.0
    
    # Perceptual loss
    MODEL.USE_PCP_LOSS = True
    MODEL.USE_STYLE_LOSS = False
    MODEL.PCP_LOSS_WEIGHT = 1.0
    MODEL.STYLE_LOSS_WEIGHT = 0
    MODEL.PCP_LOSS_TYPE = 'l1'  # l1 | l2 | fro
    MODEL.VGG_TYPE = 'vgg19'
    MODEL.VGG_LAYER_WEIGHTS = dict(conv3_4=1/8, conv4_4=1/4, conv5_4=1/2)  # before relu
    MODEL.NORM_IMG = False
    MODEL.USE_INPUT_NORM = True
    # others
    MODEL.SCALE = DATASET.SCALE
    MODEL.DOWN = 1
    MODEL.DEVICE = 'cuda'

    # solver
    SOLVER = edict()
    # generator
    SOLVER.G_OPTIMIZER = 'Adam'
    SOLVER.G_BASE_LR = 1e-4
    SOLVER.G_BETA1 = 0.9
    SOLVER.G_BETA2 = 0.999
    SOLVER.G_WEIGHT_DECAY = 0
    SOLVER.G_MOMENTUM = 0
    SOLVER.G_STEP_ITER = 1
    SOLVER.G_PREPARE_ITER = 0
    # discriminator
    SOLVER.D_OPTIMIZER = 'Adam'
    SOLVER.D_BASE_LR = 1e-4
    SOLVER.D_BETA1 = 0.9
    SOLVER.D_BETA2 = 0.999
    SOLVER.D_WEIGHT_DECAY = 0
    SOLVER.D_MOMENTUM = 0
    SOLVER.D_STEP_ITER = 1
    # both G and D
    SOLVER.WARM_UP_ITER = 2000
    SOLVER.WARM_UP_FACTOR = 0.1
    SOLVER.T_PERIOD = [200000, 400000, 150000]
    SOLVER.MAX_ITER = SOLVER.T_PERIOD[-1]

    # initialization
    CONTINUE_ITER = None
#     G_INIT_MODEL = '/root/FastMRI_challenge/temp/Simple-SR-master/RRDB_warmup.pth'
    G_INIT_MODEL = None
    D_INIT_MODEL = None

    # log and save
    LOG_PERIOD = 20
    SAVE_PERIOD = 5000

    # validation
    VAL = edict()
    VAL.PERIOD = 5000
    VAL.TYPE = 'MixDataset'
#     VAL.DATASETS = ['BSDS100']
    VAL.DATASETS = ['FASTMRI']
    VAL.SPLITS = ['VAL']
    VAL.PHASE = 'val'
    VAL.INPUT_HEIGHT = 384
    VAL.INPUT_WIDTH = 384
    VAL.SCALE = 1
    VAL.REPEAT = 1
    VAL.VALUE_RANGE = 0.001
    VAL.IMG_PER_GPU = 1
    VAL.NUM_WORKERS = 1
    VAL.SAVE_IMG = False
    VAL.TO_Y = True
    VAL.CROP_BORDER = VAL.SCALE


config = Config()



