
class My_config(object):
    log_dir = '~/FusionAttack/vis/tensorboard/fusionattack'
    physical_photo_dir = '~/FusionAttack/data/physical_photos'
    tb_logger = None
    score_thres = 0.3
    proj_scale = 50 # number of pixels corresponding to 1m
    object_ids_set = [
        # for tracking the center car in static_scenes[1]
        [
            [21], [19], [17], [15], [14],
            [11], [12], [11], [ 9], [12],
            [11], [ 9], [10], [ 8], [11],
            [ 9], [ 8], [ 8], [ 6], [ 5],
            [ 5], [ 6], [ 5], [ 4]
        ]
    ]
    static_scenes = [
        # 0: mini train set, crossing
        [
            1535489296047917, 1535489296547795, 1535489297047675, 1535489297547543, 1535489298047428,
            1535489298547291, 1535489299047201, 1535489299547057, 1535489300046941, 1535489300546798,
            1535489301047243, 1535489301547107, 1535489302046988, 1535489302547429, 1535489303047308,
            1535489303447089, 1535489303946971, 1535489304446846, 1535489304947267, 1535489305447151,
            1535489305947019, 1535489306446341, 1535489306946757, 1535489307446638, 1535489307946546,
            1535489308396618, 1535489308896501, 1535489309396386, 1535489309896252, 1535489310396680,
            1535489310946883, 1535489311396995, 1535489311897421, 1535489312397873, 1535489312897804,
            1535489313398193, 1535489313898578, 1535489314398452, 1535489314898335, 1535489315448527,
            1535489315948406
        ],
        # 1: mini train set, following
        [
            1542800855949460, 1542800856450419, 1542800856950302, 1542800857450187, 1542800857899729,
            1542800858399042, 1542800858898923, 1542800859451995, 1542800859947349, 1542800860447226,
            1542800860947086, 1542800861447555, 1542800861947962, 1542800862447900, 1542800862947705,
            1542800863447572, 1542800863948003, 1542800864448454, 1542800864948320, 1542800865448238,
            1542800865948074, 1542800866448494, 1542800866948361, 1542800867447688
        ],
    ]
    

    attack_targets = [
        #======= make sure to set `ann_file` to val set `.pkl` in model config file for following configs ===== 
        # Patch Area: top_loc, left_loc, H, W, defined for image size: [256, 704]
        # 0, Car Patch, patch 50*50 on the car (center)
        {'timestamp': 1533151610446899, 'patch_area': (76, 630, 50, 50), 'object_ids': [30]},
        # 1, Crowds Patch, patch 50*50 on the two people (center)
        {'timestamp': 1538984235447825, 'patch_area': (76, 52, 50, 50), 'object_ids': [32, 37]},


        #======= make sure to set `ann_file` to train set `.pkl` in model config file for following configs ===== 
        # 2, static scenes, crossing, road patch, untargeted, 3d in the physcial, 
        #     area definition: (H, W, alpha, dy, dx)
        {'timestamp': static_scenes[0], 'patch_area': (2, 2.5, 0, 7.3, 0), 'area_ref': 'physical'},
        # 3, static scenes, crossing, road patch, untargeted, 3d in the physcial
        #     area definition: (H, W, alpha, dy, dx)
        {'timestamp': static_scenes[0], 'patch_area': (3, 5, 0, 8, 0), 'area_ref': 'physical'},
        # 4, dynamic scenes, vehicle patch, targeted, 3d in the physcial
        #     area definition: (H, W, -1,-1,-1),  object-oriented projection
        {'timestamp': static_scenes[1], 'patch_area': (1, 1, -1, -1, -1), 'area_ref': 'physical', 'object_ids':object_ids_set[0]},
    ]
    all_objects_front = {
        1533151610446899: [10, 14, 22, 27, 43, 28, 38, 36, 40, 26, 34, 30, 37, 42],
        1538984235447825: [32, 37, 13, 17, 7, 20, 16, 24, 30, 4, 26, 34, 31, 2, 28, 35, 9, 39, 23, 5, 15, 11]
    }

if __name__ == '__main__':
    print(My_config.attack_targets[18])