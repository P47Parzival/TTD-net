import os, sys
import numpy as np
import torch
import argparse
import datetime
import wandb
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import copy

# own code
from config import Config_Generative_Model
from dataset import  create_EEG_dataset
from things_dataset import create_things_dataset
from dc_ldm.ldm_for_eeg import eLDM
from dc_ldm.sdxl_pipeline import EEGtoImageSDXL
from sc_mbm.incept_encoder import InceptSADEncoder
from eval_metrics import get_similarity_metric


def wandb_init(config, output_path):
    # wandb.init( project='dreamdiffusion',
    #             group="stageB_dc-ldm",
    #             anonymous="allow",
    #             config=config,
    #             reinit=True)
    create_readme(config, output_path)

def wandb_finish():
    wandb.finish()

def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))

def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')

def get_eval_metric(samples, avg=True):
    metric_list = ['mse', 'pcc', 'ssim', 'psm']
    res_list = []
    
    gt_images = [img[0] for img in samples]
    gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
    samples_to_run = np.arange(1, len(samples[0])) if avg else [1]
    for m in metric_list:
        res_part = []
        for s in samples_to_run:
            pred_images = [img[s] for img in samples]
            pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
            res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m)
            res_part.append(np.mean(res))
        res_list.append(np.mean(res_part))     
    res_part = []
    for s in samples_to_run:
        pred_images = [img[s] for img in samples]
        pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
        res = get_similarity_metric(pred_images, gt_images, 'class', None, 
                        n_way=50, num_trials=50, top_k=1, device='cuda')
        res_part.append(np.mean(res))
    res_list.append(np.mean(res_part))
    res_list.append(np.max(res_part))
    metric_list.append('top-1-class')
    metric_list.append('top-1-class (max)')
    return res_list, metric_list
               
def generate_images(generative_model, eeg_latents_dataset_train, eeg_latents_dataset_test, config):
    grid, _ = generative_model.generate(eeg_latents_dataset_train, config.num_samples, 
                config.ddim_steps, config.HW, 10) # generate 10 instances
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path, 'samples_train.png'))
    # wandb.log({'summary/samples_train': wandb.Image(grid_imgs)})

    grid, samples = generative_model.generate(eeg_latents_dataset_test, config.num_samples, 
                config.ddim_steps, config.HW)
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path,f'./samples_test.png'))
    for sp_idx, imgs in enumerate(samples):
        for copy_idx, img in enumerate(imgs[1:]):
            img = rearrange(img, 'c h w -> h w c')
            Image.fromarray(img).save(os.path.join(config.output_path, 
                            f'./test{sp_idx}-{copy_idx}.png'))

    # wandb.log({f'summary/samples_test': wandb.Image(grid_imgs)})

    metric, metric_list = get_eval_metric(samples, avg=config.eval_avg)
    metric_dict = {f'summary/pair-wise_{k}':v for k, v in zip(metric_list[:-2], metric[:-2])}
    metric_dict[f'summary/{metric_list[-2]}'] = metric[-2]
    metric_dict[f'summary/{metric_list[-1]}'] = metric[-1]
    # wandb.log(metric_dict)

def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img

def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)

def main(config):
    # project setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    crop_pix = int(config.crop_ratio*config.img_size)
    img_transform_train = transforms.Compose([
        normalize,

        transforms.Resize((512, 512)),
        random_crop(config.img_size-crop_pix, p=0.5),

        transforms.Resize((512, 512)),
        channel_last
    ])
    img_transform_test = transforms.Compose([
        normalize, 

        transforms.Resize((512, 512)),
        channel_last
    ])
    if config.dataset == 'EEG':
        # ── Choose dataset source ──
        dataset_type = getattr(config, 'dataset_type', 'original')
        
        if dataset_type == 'things_eeg':
            # ThingsEEG preprocessed dataset
            print("\n===== Loading ThingsEEG Dataset =====")
            eeg_latents_dataset_train, eeg_latents_dataset_test = create_things_dataset(
                processed_dir=config.things_eeg_processed_dir,
                subjects=config.things_subjects,
                image_size=config.img_size,
                test_ratio=config.things_test_ratio,
                seed=config.seed,
                augment_train=True,
                n_channels=config.things_n_channels,
                time_len=config.things_time_len,
            )
            num_voxels = config.things_time_len
            print(f"  Train: {len(eeg_latents_dataset_train)} samples")
            print(f"  Test:  {len(eeg_latents_dataset_test)} samples")
            print(f"  EEG shape: [{config.things_n_channels}, {num_voxels}]")
        else:
            # Original EEG dataset
            eeg_latents_dataset_train, eeg_latents_dataset_test = create_EEG_dataset(
                eeg_signals_path=config.eeg_signals_path,
                splits_path=config.splits_path,
                image_transform=[img_transform_train, img_transform_test],
                subject=config.subject,
            )
            num_voxels = eeg_latents_dataset_train.data_len

    else:
        raise NotImplementedError

    # prepare pretrained mbm
    pretrain_mbm_metafile = None
    if config.pretrain_mbm_path is not None:
        pretrain_mbm_metafile = torch.load(config.pretrain_mbm_path, map_location='cpu')

    # ── Create generative model (SD 1.5 or SDXL) ──
    model_type = getattr(config, 'model_type', 'sd15')
    
    if model_type == 'sdxl':
        # SDXL path: InceptSADEncoder + IPAdapterBridge + SDXL UNet
        print("\n===== Using SDXL Pipeline =====")
        # Use correct channel count for the dataset
        in_chans = getattr(config, 'things_n_channels', config.in_chans) \
                   if getattr(config, 'dataset_type', 'original') == 'things_eeg' \
                   else getattr(config, 'in_chans', 64)
        encoder = InceptSADEncoder(
            time_len=num_voxels,
            in_chans=in_chans,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            global_pool=config.global_pool,
        )
        
        generative_model = EEGtoImageSDXL(
            eeg_encoder=encoder,
            device=device,
            pretrain_root=config.pretrain_gm_path,
            sdxl_model_id=getattr(config, 'sdxl_model_id', 'stabilityai/stable-diffusion-xl-base-1.0'),
            ddim_steps=config.ddim_steps,
            global_pool=config.global_pool,
            use_clip_loss=config.clip_tune,
            pretrained_encoder_weights=pretrain_mbm_metafile.get('model', None) if pretrain_mbm_metafile else None,
        )
        
        # Resume if checkpoint exists
        if config.checkpoint_path is not None:
            ckpt = torch.load(config.checkpoint_path, map_location='cpu')
            generative_model.cond_model.load_state_dict(ckpt['cond_model'])
            generative_model.unet.load_state_dict(ckpt['unet'])
            print('SDXL model resumed from checkpoint')
        
        # Fine-tune
        generative_model.finetune(
            eeg_latents_dataset_train, eeg_latents_dataset_test,
            config, config.output_path
        )
        
        # Generate images
        grid, samples = generative_model.generate(
            eeg_latents_dataset_test,
            num_samples=config.num_samples,
            ddim_steps=config.ddim_steps,
            limit=50,
            output_path=os.path.join(config.output_path, 'eval')
        )
    else:
        # Original SD 1.5 path
        print("\n===== Using SD 1.5 Pipeline =====")
        generative_model = eLDM(pretrain_mbm_metafile, num_voxels,
                    device=device, pretrain_root=config.pretrain_gm_path, logger=config.logger, 
                    ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond, clip_tune = config.clip_tune, cls_tune = config.cls_tune)
        
        # resume training if applicable
        if config.checkpoint_path is not None:
            model_meta = torch.load(config.checkpoint_path, map_location='cpu')
            generative_model.model.load_state_dict(model_meta['model_state_dict'])
            print('model resumed')
        # finetune the model
        trainer = create_trainer(config.num_epoch, config.precision, config.accumulate_grad, config.logger, check_val_every_n_epoch=2)
        generative_model.finetune(trainer, eeg_latents_dataset_train, eeg_latents_dataset_test,
                    config.batch_size, config.lr, config.output_path, config=config)

        # generate images
        generate_images(generative_model, eeg_latents_dataset_train, eeg_latents_dataset_test, config)

    return

def get_args_parser():
    parser = argparse.ArgumentParser('Double Conditioning LDM Finetuning', add_help=False)
    # project parameters
    parser.add_argument('--seed', type=int)
    parser.add_argument('--root_path', type=str, default = '../dreamdiffusion/')
    parser.add_argument('--pretrain_mbm_path', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--crop_ratio', type=float)
    parser.add_argument('--dataset', type=str)

    # finetune parameters
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--precision', type=int)
    parser.add_argument('--accumulate_grad', type=int)
    parser.add_argument('--global_pool', type=bool)

    # diffusion sampling parameters
    parser.add_argument('--pretrain_gm_path', type=str)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--ddim_steps', type=int)
    parser.add_argument('--use_time_cond', type=bool)
    parser.add_argument('--eval_avg', type=bool)

    # # distributed training parameters
    # parser.add_argument('--local_rank', type=int)

    return parser

def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)


def create_trainer(num_epoch, precision=32, accumulate_grad_batches=2,logger=None,check_val_every_n_epoch=0):
    acc = 'gpu' if torch.cuda.is_available() else 'cpu'
    return pl.Trainer(accelerator=acc, max_epochs=num_epoch, logger=logger, 
            precision=precision, accumulate_grad_batches=accumulate_grad_batches,
            enable_checkpointing=False, enable_model_summary=False, gradient_clip_val=0.5,
            check_val_every_n_epoch=check_val_every_n_epoch)
  
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = Config_Generative_Model()
    config = update_config(args, config)
    
    if config.checkpoint_path is not None:
        model_meta = torch.load(config.checkpoint_path, map_location='cpu')
        ckp = config.checkpoint_path
        config = model_meta['config']
        config.checkpoint_path = ckp
        print('Resuming from checkpoint: {}'.format(config.checkpoint_path))

    output_path = os.path.join(config.output_path, 'results', 'generation',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    config.output_path = output_path
    os.makedirs(output_path, exist_ok=True)
    
    wandb_init(config, output_path)

    # logger = WandbLogger()
    config.logger = None # logger
    main(config)
