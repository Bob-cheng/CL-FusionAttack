import torch


class MaskWeightUpdater():
    def __init__(self, initweight, maskloss_thresh, total_steps, upscaler=1.5,  downscaler=0.7, interval=20) -> None:
        self.upscaler = upscaler
        self.downscaler = downscaler
        self.interval = interval
        self.init_ratio = 1
        self.total_steps = total_steps
        self.current_step = 0
        self.maskloss_thresh  = maskloss_thresh
        self.mask_weight = initweight

    def get_target_ratio(self):
        m = self.init_ratio
        f = 0.6 * self.total_steps
        g = self.maskloss_thresh
        x = self.current_step
        if self.current_step > f:
            return self.maskloss_thresh
        else:
            return (m-g)/(f*f)*x*x + 2 * (g-m)/f * x + m

    
    def step(self, mask_ratio):
        ref_value = mask_ratio.item()
        self.current_step += 1
        if self.current_step % self.interval == 0:
            target_ratio = self.get_target_ratio() # square
            print("target_ratio:", target_ratio, "current_ratio", ref_value)
            if ref_value >= target_ratio:
                self.mask_weight *= self.upscaler
            else:
                self.mask_weight *= self.downscaler

        return self.mask_weight

    def get_mask_weight(self):
        return self.mask_weight

    def get_mask_ratio(self, mask_shape, mask_init):
        H, W = mask_shape
        ratio = (mask_init[1] - mask_init[0]) * (mask_init[3] - mask_init[2]) / (H * W)
        return ratio