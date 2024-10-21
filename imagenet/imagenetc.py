import logging

import torch
import torch.optim as optim

from robustbench.data import load_imagenetc, generate_cdc_order, load_imagenetc_custom
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

import tent
import cotta
import vida

from conf import cfg, load_cfg_fom_args

from vpt import PromptViT
from dpcore import DPCore

import pdb

import webdataset as wds
import torchvision.transforms as trn
import os

from tqdm import tqdm 

logger = logging.getLogger(__name__)

def set_model(description):
    args = load_cfg_fom_args(description)
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()

    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    if cfg.MODEL.ADAPTATION == "cotta":
        logger.info("test-time adaptation: CoTTA")
        model = setup_cotta(base_model)
    if cfg.MODEL.ADAPTATION == "vida":
        logger.info("test-time adaptation: ViDA")
        model = setup_vida(args, base_model)
    if cfg.MODEL.ADAPTATION == "dpcore":
        logger.info("test-time adaptation: DPCore")
        model = setup_dpcore(args, base_model)

    model_name = cfg.MODEL.ADAPTATION
    return model, model_name

def evaluate_csc(description):
    logger.info("CSC Setting")
    
    model, model_name = set_model(description)
    
    num_rounds = 10
    round_error = []
    for round in range(1, num_rounds+1):
        All_error = []
        for ii, severity in enumerate(cfg.CORRUPTION.SEVERITY):
            for i_x, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
                # reset adaptation for each combination of corruption x severity
                # note: for evaluation protocol, but not necessarily needed
                try:
                    if i_x == 0 and round == 1:
                        if hasattr(model, 'rest'):
                            model.reset()
                            logger.info("resetting model")
                    else:
                        pass
                        # logger.warning("not resetting model")
                except:
                    logger.warning("not resetting model")


                # measure elan error after adapting to each corruption
                test_loader = load_imagenetc_custom(cfg.TEST.BATCH_SIZE,
                                            severity, cfg.DATA_DIR, True,
                                                        [corruption_type])
                total_samples = 0
                acc = 0.
                with torch.no_grad():
                    for batch_idx, (x_test, y_test, _) in enumerate(test_loader):
                        x_test, y_test = x_test.cuda(), y_test.cuda()

                        output = model(x_test) 
                        output = output[0] if isinstance(output, tuple) else output
                        acc += (output.max(1)[1] == y_test).float().sum()
                        cur_acc = (output.max(1)[1] == y_test).float().sum() / x_test.shape[0]

                        total_samples += x_test.shape[0]

                        logger.info(f"[{corruption_type}/{severity}], Current batch {batch_idx} acc % {cur_acc:.3%}")

                acc = acc.item() / total_samples
                err = 1. - acc
                All_error.append(err)

                logger.info(f"Round: {round}, error % [{corruption_type}{severity}]: {err:.2%}"
                            f"{', coreset size ' + str(len(model.coreset)) if model_name == 'dpcore' else ''}")

        all_error_res = ' '.join([f"{e:.2%}" for e in All_error])
        logger.info(f"Round: {round}, All error: {all_error_res}")
        logger.info(f"Round: {round}, Mean error: {sum(All_error) / len(All_error):.2%}")
        if model_name == 'dpcore':
            logger.info(f"Round: {round}, Coreset size: {len(model.coreset)}")
        round_error.append(sum(All_error) / len(All_error))

    all_error_res = ' '.join([f"{e:.2%}" for e in round_error])
    logger.info(f"All Round Error: {all_error_res}")
    

def identity(x):
    return x

def get_webds_loader(dset_name):
    #url = os.path.join(dset_path, "serial_{{00000..99999}}.tar") Uncoment this to use a local copy of CCC
    url = f'https://mlcloud.uni-tuebingen.de:7443/datasets/CCC/{dset_name}/serial_{{00000..99999}}.tar'

    normalize = trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preproc = trn.Compose(
        [
            trn.ToTensor(),
            normalize,
        ]
    )
    dataset = (
        wds.WebDataset(url)
        .decode("pil")
        .to_tuple("input.jpg", "output.cls")
        .map_tuple(preproc, identity)
    )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=2, batch_size=64)
    return dataloader

def evaluate_ccc(description):
    logger.info("CCC Setting")
    
    model, model_name = set_model(description)

    # baseline acc: 0, 20, 40
    # processind  |  cur_seed  |  speed
    # ---------------------------------
    #     0       |     43     |  1000
    #     1       |     44     |  1000
    #     2       |     45     |  1000
    #     3       |     43     |  2000
    #     4       |     44     |  2000
    #     5       |     45     |  2000
    #     6       |     43     |  5000
    #     7       |     44     |  5000
    #     8       |     45     |  5000

    baseline = 20
    processind = 7
    cur_seed = [43, 44, 45][processind % 3]
    speed = [1000, 2000, 5000][int(processind / 3)]

    # logs = "/root/DPCore/imagenet/output"
    # exp_name = "ccc_{}".format(str(baseline))

    # if not os.path.exists(os.path.join(logs, exp_name)):
    #     os.mkdir(os.path.join(logs, exp_name))

    # file_name = os.path.join(
    #     logs,
    #     exp_name,
    #     "model_{}_baseline_{}_transition+speed_{}_seed_{}.txt".format(
    #         str(model_name), str(baseline), str(speed), str(cur_seed)
    #     ),
    # )

    dset_name = "baseline_{}_transition+speed_{}_seed_{}".format(
        str(baseline), str(speed), str(cur_seed)
    )
    total_seen_so_far = 0
    dataset_loader = get_webds_loader(dset_name)

    logger.info(dset_name)

    if hasattr(model, 'rest'):
        model.reset()
        logger.info("resetting model")

    for i, (images, labels) in enumerate(dataset_loader):
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.no_grad():
            output = model(images)
        output = output[0] if isinstance(output, tuple) else output

        num_images_in_batch = images.size(0)
        total_seen_so_far += num_images_in_batch

        vals, pred = (output).max(dim=1, keepdim=True)
        correct_this_batch = pred.eq(labels.view_as(pred)).sum().item()

        log_message = "# os {}, acc = {:.10f}".format(
            total_seen_so_far,
            float(100 * correct_this_batch) / images.size(0)
        )

        # Check if the model name is 'dpcore'
        if model_name == 'dpcore': 
            coreset_size = len(model.coreset)  # Get coreset size
            log_message += ", coreset size = {}".format(coreset_size)  # Append coreset size

        logger.info(log_message)

        if total_seen_so_far > 7500000:
            return

def evaluate_cdc(description):
    logger.info("CDC Setting")

    model, model_name = set_model(description)

    All_error = []
    corruption_error = {}

    corruptions = cfg.CORRUPTION.TYPE
    num_total_batches = cfg.CORRUPTION.NUM_EX // cfg.TEST.BATCH_SIZE + 1
    cdc_domain_order = generate_cdc_order(corruptions, num_total_batches)

    for ii, severity in enumerate(cfg.CORRUPTION.SEVERITY):
        domain_iters = {}
        for i_x, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            # reset adaptation for each combination of corruption x severity
            # note: for evaluation protocol, but not necessarily needed
            try:
                if i_x == 0:
                    if hasattr(model, 'rest'):
                        model.reset()
                        logger.info("resetting model")
                else:
                    logger.warning("not resetting model")
            except:
                logger.warning("not resetting model")

            domain_data_loader = load_imagenetc_custom(cfg.TEST.BATCH_SIZE,
                                           severity, cfg.DATA_DIR, True,
                                           [corruption_type])
            domain_iters[corruption_type] = iter(domain_data_loader)
            corruption_error[corruption_type] = []

        for domain in tqdm(cdc_domain_order, desc="Processing batches", unit="batches"):
            data_iter = domain_iters[domain]

            x_test, y_test, _ = next(data_iter)

            x_test, y_test = x_test.cuda(), y_test.cuda()
            
            acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE)
            err = 1. - acc
            All_error.append(err)
            
            corruption_error[domain].append(err)
            if model_name == 'dpcore':
                logger.info(f"{domain}: {err:.2%}, size of coreset {len(model.coreset)}")

        for corruption_type, error in corruption_error.items():
            logger.info(f"error % [{corruption_type}{severity}]: {sum(error) / len(error):.2%}")

    logger.info(f"Mean error: {sum(All_error) / len(All_error):.2%}")

def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model

def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model

def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=0.9,
                   dampening=0,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=True)
    else:
        raise NotImplementedError

def setup_cotta(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model

def setup_vida(args, model):
    model = vida.configure_model(model, cfg)
    model_param, vida_param = vida.collect_params(model)
    optimizer = setup_optimizer_vida(model_param, vida_param, cfg.OPTIM.LR, cfg.OPTIM.ViDALR)
    vida_model = vida.ViDA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           unc_thr = args.unc_thr,
                           ema = cfg.OPTIM.MT,
                           ema_vida = cfg.OPTIM.MT_ViDA,
                           )
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return vida_model

def setup_optimizer_vida(params, params_vida, model_lr, vida_lr):
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam([{"params": params, "lr": model_lr},
                                  {"params": params_vida, "lr": vida_lr}],
                                 lr=1e-5, betas=(cfg.OPTIM.BETA, 0.999),weight_decay=cfg.OPTIM.WD)

    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD([{"params": params, "lr": model_lr},
                                  {"params": params_vida, "lr": vida_lr}],
                                    momentum=cfg.OPTIM.MOMENTUM,dampening=cfg.OPTIM.DAMPENING,
                                    nesterov=cfg.OPTIM.NESTEROV,
                                 lr=1e-5,weight_decay=cfg.OPTIM.WD)
    else:
        raise NotImplementedError
    
def setup_dpcore(args, model):
    net = PromptViT(model, 4).cuda()
    adapt_model = DPCore(net)
    logger.info(f'{args}')

    #TODO: 要重新计算train_info
    # 需要source data写一个train_loader
    # 先给一个fake_src_stat,保存路径记得修改
    # import torchvision.transforms as transforms
    # from robustbench.loaders import CustomImageFolder
    # import torch.utils.data as data
    # data_folder_path = "/root/autodl-tmp/data/imagenetc/brightness/5"
    # trainsforms_test = transforms.Compose([transforms.Resize(256),
    #                                      transforms.CenterCrop(224),
    #                                      transforms.ToTensor()])
    # train_dataset = CustomImageFolder(data_folder_path, trainsforms_test)
    # train_loader = data.DataLoader(train_dataset, batch_size=100, shuffle=False,
    #                                num_workers=2)

    adapt_model.obtain_src_stat()

    return adapt_model

if __name__ == '__main__':
    evaluate_csc('"Imagenet-C evaluation.')
