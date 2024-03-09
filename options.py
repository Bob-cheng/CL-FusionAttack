import argparse

def parse():
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('model_name', type=str,
                            choices=['bevfusion', 'bevfusion2', 'uvtr', 'deepint', 'transfusion', 'bevformer'] ,help="the model to attack.")
    parser.add_argument('run_type', type=str, 
                            choices=['train', 'eval', 'both', 'eval_all', 'train_all'], help='The running type of the program')
    parser.add_argument('--patch_cfg', type=int, required=True, help='the index of the patch config.')
    parser.add_argument('--n_iters', type=int, default=5000, help='maximim iterations to try')
    parser.add_argument('--loss_type', type=str, default='log_score_loss', 
                            choices=['log_score_loss', 'score_loss', 'all_loss', 'score_loss_FP', 'dense_heatmap_loss'], help='step of each attack')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size of different transformations')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--patch_type', type=str, default='rec', 
                            choices=['rec', 'whole', 'dynamic'], help='The type of the patch to generate')
    parser.add_argument('--adp_thres', type=float, default=-1., 
                            help='adaptive optimization for sensitivity optimization. negative values indicate no adaptive.')
    parser.add_argument('--mask_step', type=int, default=2, help='the mask granulity for sensitivity optimization.')
    parser.add_argument('--mask_weight', type=float, default=1., help='the initial mask loss weight')
    parser.add_argument('--mask_lr', type=float, default=0.1, help='learning rate of dynamic mask optimization.')
    parser.add_argument('--obj_type', type=str, default='None', 
                            choices=['None', 'Targeted', 'Front'], help='The type of target objects.')
    parser.add_argument('--test_name', type=str, help='The name of this test.')
    parser.add_argument('--trans', action='store_true', help='whether we do EoT?')
    parser.add_argument('--tv_loss', action='store_true', help='whether we use tv_loss?')
    parser.add_argument('--score_tres', type=float, help='score threshold of drawing bboxes in log')
    parser.add_argument('--patch_fid', type=int, default=50, help='the fidelity of adversarial patch')

    args = parser.parse_args()
    return vars(args)