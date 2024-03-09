import os
import torch
from my_config import My_config
from model_loader import Model_loader
from tensorboardX import SummaryWriter
from my_utils import sample_train_val
from model_evaluater import Model_evaluater
from fusion_attacker import FusionAttacker
from patch_converter import Patch_converter
import options

def eval_main(model_name,  patch_area,  attack_timestamp, object_ids, args):
    model_loader = Model_loader()
    
    target_model, target_dataloader, target_cfg = model_loader.load_model(model_name)
    trans_evaluater = Model_evaluater(model_name, target_model, target_dataloader, target_cfg)
    
    best_patch = torch.load(os.path.join(My_config.log_dir, "bevfusion_patch_latest.pt"))
    # best_patch = torch.load(os.path.join(My_config.log_dir, "bevfusion_patch.pt"))
    mask = torch.load(os.path.join(My_config.log_dir, "bevfusion_mask.pt"))

    os.makedirs(os.path.join(My_config.log_dir, 'ben_imgs'), exist_ok=True)
    os.makedirs(os.path.join(My_config.log_dir, 'ben_lidar'), exist_ok=True)
    os.makedirs(os.path.join(My_config.log_dir, 'adv_imgs'), exist_ok=True)
    os.makedirs(os.path.join(My_config.log_dir, 'adv_lidar'), exist_ok=True)
    
    trans_evaluater.eval_patch(best_patch, mask, attack_timestamp, patch_area, args,\
                            object_ids=object_ids)

    
def fusion_attack_main(model_names, patch_area, attack_timestamp, object_ids, args):
    fusion_attacker = FusionAttacker(model_names)
    fusion_attacker.attack(attack_timestamp, patch_area, object_ids, args)

    # save patch and mask
    torch.save(fusion_attacker.best_patch, os.path.join(My_config.log_dir, "bevfusion_patch.pt"))
    torch.save(fusion_attacker.mask, os.path.join(My_config.log_dir, "bevfusion_mask.pt"))



if __name__ == "__main__":
    args = options.parse()
    print(args)
    pc = Patch_converter()
    My_config.proj_scale = args['patch_fid']
    if args['score_tres'] is not None:
        My_config.score_thres = args['score_tres']
    print(f"log bbox threshold: {My_config.score_thres}")
    # choose attack scene
    atk_target          = My_config.attack_targets[args['patch_cfg']]
    patch_area          = atk_target['patch_area'] # top_loc, left_loc, H, W
    args['replace']     = atk_target['replace'] if 'replace' in atk_target.keys() else None
    if 'area_ref' in atk_target.keys():
        area_ref = atk_target['area_ref']
    else:
        area_ref = 'bevfusion'
    if area_ref != 'physical':
        patch_area = pc.convert_patch_area(patch_area, target='bevfusion', source=area_ref)
    target_object_ids   = atk_target['object_ids'] if 'object_ids' in atk_target.keys() else None
    attack_timestamp    = atk_target['timestamp']
    if type(attack_timestamp) is list:
        if args['run_type'] == 'train':
            attack_timestamp, target_object_ids = sample_train_val(attack_timestamp, target_object_ids)[:2]
        elif args['run_type'] == 'eval':
            attack_timestamp, target_object_ids = sample_train_val(attack_timestamp, target_object_ids)[2:]
        elif args['run_type'] == 'eval_all' or args['run_type'] == 'train_all':
            attack_timestamp, target_object_ids = attack_timestamp, target_object_ids
        front_object_ids = None
    else:
        if attack_timestamp in My_config.all_objects_front.keys():
            front_object_ids = My_config.all_objects_front[attack_timestamp]
        else:
            front_object_ids = None
    
    if args['obj_type'] == 'None':
        train_object_ids = None
    elif args['obj_type'] == 'Targeted':
        train_object_ids = target_object_ids
    elif args['obj_type'] == 'Front':
        train_object_ids = front_object_ids
    print(f"timestamp: {attack_timestamp}, train object_ids: {train_object_ids}, replace dict: {args['replace']}")


    # prepare log_dir and logger
    patch_type = args['patch_type']
    if patch_type == 'whole' or patch_type == 'dynamic':
        postfix = '{}-b{}'.format(patch_type, 'front_GT' if train_object_ids is not None else 'None')
    elif patch_type == 'rec':
        postfix = "{}-{}-{}-{}-b{}".format(
            patch_area[0],
            patch_area[1],
            patch_area[2],
            patch_area[3],
            str(train_object_ids) if train_object_ids is None or type(train_object_ids[0]) is int else 'Targeted'
        )
    
    if  args['run_type'][:4] == 'eval':
        test_name = args['test_name']
    else:
        test_name = "{}-{}-{}".format(
            args['test_name'], 
            attack_timestamp if type(attack_timestamp) is int else f"mycfg_{args['patch_cfg']}", 
            postfix
        )
    

    My_config.log_dir = os.path.join(My_config.log_dir, test_name)
    os.makedirs(My_config.log_dir, exist_ok=True)
    My_config.tb_logger = SummaryWriter(My_config.log_dir)
    print("The log dir: " + My_config.log_dir)
    My_config.tb_logger.add_text('CLI_args', str(args), 0)

    if args['run_type'][:5] == 'train' or args['run_type'] == 'both' :
        fusion_attack_main([args['model_name']], patch_area, attack_timestamp, train_object_ids, args)
    elif args['run_type'][:4] == 'eval' or args['run_type'] == 'both':
        eval_main(args['model_name'],  patch_area, attack_timestamp, train_object_ids, args)

