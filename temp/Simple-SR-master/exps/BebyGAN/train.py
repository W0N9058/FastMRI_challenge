import argparse
import os
import os.path as osp
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/../..')
import logging
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from config import config
from utils import common, dataloader, solver, model_opr, resizer, region_seperator
from dataset import get_dataset
from network import Network
from validate import validate


def init_dist(local_rank):
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method='env://')
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    rank = 0
    num_gpu = 1
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        num_gpu = int(os.environ['WORLD_SIZE'])
        distributed = num_gpu > 1
    if distributed:
        rank = args.local_rank
        init_dist(rank)
    common.init_random_seed(config.DATASET.SEED)
    print(rank)

    exp_dir, cur_dir = osp.split(osp.split(osp.realpath(__file__))[0])
    root_dir = osp.split(exp_dir)[0]
    log_dir = osp.join(root_dir, 'logs', cur_dir)
    model_dir = osp.join(log_dir, 'models')
    solver_dir = osp.join(log_dir, 'solvers')
    if rank <= 0:
        common.mkdir(log_dir)
        ln_log_dir = osp.join(exp_dir, cur_dir, 'log')
        if not osp.exists(ln_log_dir):
            os.system('ln -s %s log' % log_dir)
        common.mkdir(model_dir)
        common.mkdir(solver_dir)
        save_dir = osp.join(log_dir, 'saved_imgs')
        common.mkdir(save_dir)
        tb_dir = osp.join(log_dir, 'tb_log')
        tb_writer = SummaryWriter(tb_dir)
        common.setup_logger('base', log_dir, 'train', level=logging.INFO, screen=True, to_file=True)
        logger = logging.getLogger('base')

    train_dataset = get_dataset(config.DATASET)
    train_loader = dataloader.train_loader(train_dataset, config, rank=rank, seed=config.DATASET.SEED, is_dist=distributed)
    if rank <= 0:
        val_dataset = get_dataset(config.VAL)
        val_loader = dataloader.val_loader(val_dataset, config, rank, 1)

    model = Network(config)
    if rank <= 0:
        print(model.G)
        print(model.D)

    if config.G_INIT_MODEL:
        if rank <= 0:
            logger.info('[Initing] Generator')
        model_opr.load_model(model.G, config.G_INIT_MODEL, strict=False, cpu=True)
    if config.D_INIT_MODEL:
        if rank <= 0:
            logger.info('[Initing] Discriminator')
        model_opr.load_model(model.D, config.D_INIT_MODEL, strict=False, cpu=True)

    if config.CONTINUE_ITER:
        if rank <= 0:
            logger.info('[Loading] Iter: %d' % config.CONTINUE_ITER)
        G_model_path = osp.join(model_dir, '%d_G.pth' % config.CONTINUE_ITER)
        D_model_path = osp.join(model_dir, '%d_D.pth' % config.CONTINUE_ITER)
        model_opr.load_model(model.G, G_model_path, strict=True, cpu=True)
        model_opr.load_model(model.D, D_model_path, strict=True, cpu=True)

    device = torch.device(config.MODEL.DEVICE)
    model.set_device(device)
    if distributed:
        model.G = torch.nn.parallel.DistributedDataParallel(model.G, device_ids=[torch.cuda.current_device()])
        model.D = torch.nn.parallel.DistributedDataParallel(model.D, device_ids=[torch.cuda.current_device()])

    G_optimizer = solver.make_optimizer_sep(model.G, config.SOLVER.G_BASE_LR, type=config.SOLVER.G_OPTIMIZER,
                                            beta1=config.SOLVER.G_BETA1, beta2=config.SOLVER.G_BETA2,
                                            weight_decay=config.SOLVER.G_WEIGHT_DECAY,
                                            momentum=config.SOLVER.G_MOMENTUM,
                                            num_gpu=None)
    G_lr_scheduler = solver.CosineAnnealingLR_warmup(config, G_optimizer, config.SOLVER.G_BASE_LR)
    D_optimizer = solver.make_optimizer_sep(model.D, config.SOLVER.D_BASE_LR, type=config.SOLVER.D_OPTIMIZER,
                                            beta1=config.SOLVER.D_BETA1, beta2=config.SOLVER.D_BETA2,
                                            weight_decay=config.SOLVER.D_WEIGHT_DECAY,
                                            momentum=config.SOLVER.D_MOMENTUM,
                                            num_gpu=None)
    D_lr_scheduler = solver.CosineAnnealingLR_warmup(config, D_optimizer, config.SOLVER.D_BASE_LR)

    iteration = 0
    if config.CONTINUE_ITER:
        G_solver_path = osp.join(solver_dir, '%d_G.solver' % config.CONTINUE_ITER)
        iteration = model_opr.load_solver(G_optimizer, G_lr_scheduler, G_solver_path)
        D_solver_path = osp.join(solver_dir, '%d_D.solver' % config.CONTINUE_ITER)
        _ = model_opr.load_solver(D_optimizer, D_lr_scheduler, D_solver_path)

    scaler = torch.cuda.amp.GradScaler()  # Mixed Precision 사용을 위한 GradScaler 초기화

    max_iter = max_psnr = max_ssim = 0
    for lr_img, hr_img in train_loader:
        iteration = iteration + 1
    
        model.G.train()
        model.D.train()
    
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)
    
        # loss_dict 초기화
        loss_dict = OrderedDict()
    
        flat_mask = region_seperator.get_flat_mask(lr_img, kernel_size=config.MODEL.FLAT_KSIZE,
                                                   std_thresh=config.MODEL.FLAT_STD,
                                                   scale=config.DATASET.SCALE)
    
        for p in model.D.parameters():
            p.requires_grad = False
    
        G_optimizer.zero_grad()
    
        with torch.cuda.amp.autocast():  # Mixed Precision을 위해 autocast 사용
            output = model.G(lr_img)
            output_det = output * (1 - flat_mask)
            hr_det = hr_img * (1 - flat_mask)
            bp_lr_img = resizer.imresize(output, scale=1/config.DATASET.SCALE)
    
            gen_loss = 0.0
            recon_loss = model.recon_loss_weight * model.recon_criterion(output, hr_img)
            loss_dict['G_REC'] = recon_loss  # 추가: 재구성 손실을 loss_dict에 추가
            gen_loss += recon_loss
    
            bp_loss = model.bp_loss_weight * model.bp_criterion(bp_lr_img, lr_img)
            loss_dict['G_BP'] = bp_loss  # 추가: 백 프로젝션 손실을 loss_dict에 추가
            gen_loss += bp_loss
    
            if iteration > config.SOLVER.G_PREPARE_ITER:
                if model.use_pcp:
                    pcp_loss, style_loss, _ = model.pcp_criterion(output, hr_img)
                    loss_dict['G_PCP'] = model.pcp_loss_weight * pcp_loss  # 추가: PCP 손실을 loss_dict에 추가
                    gen_loss += model.pcp_loss_weight * pcp_loss
                    if style_loss is not None:
                        loss_dict['G_STY'] = model.style_loss_weight * style_loss  # 추가: 스타일 손실을 loss_dict에 추가
                        gen_loss += model.style_loss_weight * style_loss
    
                gen_real = model.D(hr_det).detach()
                gen_fake = model.D(output_det)
                gen_real_loss = model.adv_criterion(gen_real - torch.mean(gen_fake), False, is_disc=False) * 0.5
                gen_fake_loss = model.adv_criterion(gen_fake - torch.mean(gen_real), True, is_disc=False) * 0.5
                gen_adv_loss = model.adv_loss_weight * (gen_real_loss + gen_fake_loss)
                loss_dict['G_ADV'] = gen_adv_loss  # 추가: 생성자 적대적 손실을 loss_dict에 추가
                gen_loss += gen_adv_loss
    
        scaler.scale(gen_loss).backward()
        scaler.step(G_optimizer)
        scaler.update()
        G_lr_scheduler.step()
    
        if iteration % config.SOLVER.D_STEP_ITER == 0 and iteration > config.SOLVER.G_PREPARE_ITER:
            for p in model.D.parameters():
                p.requires_grad = True
            D_optimizer.zero_grad()
    
            with torch.cuda.amp.autocast():
                dis_fake = model.D(output_det).detach()
                dis_real = model.D(hr_det)
                dis_real_loss = model.adv_criterion(dis_real - torch.mean(dis_fake), True, is_disc=True) * 0.5
                dis_fake = model.D(output_det.detach())
                dis_fake_loss = model.adv_criterion(dis_fake - torch.mean(dis_real.detach()), False, is_disc=True) * 0.5
    
                # 손실 값을 loss_dict에 추가
                loss_dict['D_ADV'] = dis_real_loss + dis_fake_loss
    
            scaler.scale(loss_dict['D_ADV']).backward()
            scaler.step(D_optimizer)
            scaler.update()
            D_lr_scheduler.step()
    
        torch.cuda.empty_cache()  # 메모리 캐시 비우기
    
        if rank <= 0:
            if iteration % config.LOG_PERIOD == 0 or iteration == config.SOLVER.MAX_ITER:
                log_str = f'Iter: {iteration}, LR: {G_optimizer.param_groups[0]["lr"]:.3e}, '
                for key, value in loss_dict.items():
                    tb_writer.add_scalar(key, value.mean(), global_step=iteration)
                    log_str += f'{key}: {value:.6f}, '
                logger.info(log_str)
    
            if iteration % config.SAVE_PERIOD == 0 or iteration == config.SOLVER.MAX_ITER:
                logger.info(f'[Saving] Iter: {iteration}')
                model_opr.save_model(model.G, osp.join(model_dir, f'{iteration}_G.pth'))
                model_opr.save_model(model.D, osp.join(model_dir, f'{iteration}_D.pth'))
                model_opr.save_solver(G_optimizer, G_lr_scheduler, iteration, osp.join(solver_dir, f'{iteration}_G.solver'))
                model_opr.save_solver(D_optimizer, D_lr_scheduler, iteration, osp.join(solver_dir, f'{iteration}_D.solver'))
    
            if iteration % config.VAL.PERIOD == 0 or iteration == config.SOLVER.MAX_ITER:
                logger.info(f'[Validating] Iter: {iteration}')
                model.G.eval()
                with torch.no_grad():
                    psnr, ssim = validate(model, val_loader, config, device, iteration, save_path=save_dir)
                if psnr > max_psnr:
                    max_psnr, max_ssim, max_iter = psnr, ssim, iteration
                logger.info(f'[Val Result] Iter: {iteration}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}')
                logger.info(f'[Best Result] Iter: {max_iter}, PSNR: {max_psnr:.4f}, SSIM: {max_ssim:.4f}')
    
        if iteration >= config.SOLVER.MAX_ITER:
            break
    
    if rank <= 0:
        logger.info('Finish training process!')
        logger.info(f'[Final Best Result] Iter: {max_iter}, PSNR: {max_psnr:.4f}, SSIM: {max_ssim:.4f}')

if __name__ == '__main__':
    main()