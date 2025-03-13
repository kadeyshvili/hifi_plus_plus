import torch.nn as nn
from src.model.dynamic_upsampling_generator import A2AHiFiPlusGeneratorV4
from src.model.discriminator_p import MultiPeriodDiscriminator
from src.model.discriminator_s import MultiScaleDiscriminator


class HiFiGAN(nn.Module):
    def __init__(self,
                 generator_config,
                 mpd_config,
                 msd_config):
        super().__init__()
        self.generator = A2AHiFiPlusGeneratorV4(**generator_config)
        self.mpd = MultiPeriodDiscriminator(**mpd_config)
        self.msd = MultiScaleDiscriminator(**msd_config)


    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        return result_info