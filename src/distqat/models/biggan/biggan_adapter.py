from locale import D_FMT
import torch
import functools
from distqat.models.biggan.biggan import Generator, Discriminator, G_D
from distqat.models.biggan.utils import ema, prepare_z_y, toggle_grad, seed_rng
from distqat.models.biggan import losses
from distqat.models.biggan import inception_utils
from hivemind.moe.server.layers.custom_experts import register_expert_class
import logging
from pathlib import Path

# _cfg = parse_yaml_file_as(Config, "configs/biggan_full.yaml")

def head_sample_input(batch_size, num_channels: int, img_size: int):
    return (
    # (torch.empty((batch_size, 3, 32, 32)), torch.empty((batch_size), dtype=torch.long))
    torch.empty((batch_size, num_channels, img_size, img_size)), 
    torch.empty((batch_size), dtype=torch.long)
    )

logger = logging.getLogger(__name__)

class InnerGANOptimizer(torch.optim.Optimizer):
    """
    Composite optimizer that wraps two real optimizers (D, G)
    and exposes a single-optimizer interface for DiLoCo.
    """
    def __init__(self, param_groups, g_opt_ctor, g_opt_kwargs, d_opt_ctor, d_opt_kwargs):
        # Torch wants defaults; we don't use them here.
        super().__init__(params=param_groups, defaults={})
        # Split by role
        g_groups = [pg for pg in param_groups if pg.get("role") == "G"]
        d_groups = [pg for pg in param_groups if pg.get("role") == "D"]
        if not g_groups or not d_groups:
            raise ValueError("Expected param_groups with 'role'=='G' and 'role'=='D'.")

        # Build real optimizers on the *same* group objects
        self.optim_G = g_opt_ctor(g_groups, **g_opt_kwargs)
        self.optim_D = d_opt_ctor(d_groups, **d_opt_kwargs)

        # Expose a unified view (DiLoCo iterates .param_groups to find params)
        self.param_groups = []
        self.param_groups.extend(self.optim_G.param_groups)
        self.param_groups.extend(self.optim_D.param_groups)

        # Expose a merged state dict if someone inspects it (optional)
        self.state = {}
        self.state.update(self.optim_G.state)
        self.state.update(self.optim_D.state)

    @torch.no_grad()
    def step(self, *args, **kwargs):
        # Step order for GAN inner loop: D first, then G (typical)

        self.optim_D.step()
        self.optim_G.step()

    def zero_grad(self, set_to_none: bool = False):
        self.optim_D.zero_grad(set_to_none=set_to_none)
        self.optim_G.zero_grad(set_to_none=set_to_none)

    # Optional: checkpointing support (can add later)
    def state_dict(self):
        return {
            "G": self.optim_G.state_dict(),
            "D": self.optim_D.state_dict(),
        }

    def load_state_dict(self, sd):
        self.optim_G.load_state_dict(sd["G"])
        self.optim_D.load_state_dict(sd["D"])


@register_expert_class("biggan.full", head_sample_input)
class BigGANAdapter(torch.nn.Module):
    def __init__(self, *args, num_classes: int, config: dict, **kwargs):
        """
        Initialize BigGAN adapter.
        
        Args:
            num_classes: Number of classes for the dataset
            config: Dictionary containing BigGAN-specific configuration parameters
        """
        
        # biggan is already a dict from kwargs_from_config
        super().__init__()
        self.device = config['device']
        self.batch_size = config['batch_size']
        self.config = config
        self.G = Generator(**config).to(self.device)
        self.D = Discriminator(**config).to(self.device)
        self.GD = G_D(self.G, self.D)

        if config['ema']:
            self.G_ema = Generator(**{**(config), 'skip_init':True, 'no_optim':True}).to(self.device)
            self.ema = ema(self.G, self.G_ema, config['ema_decay'], config['ema_start'])
        else:
            self.G_ema = None
            self.ema = None

        seed_rng(0)
        torch.backends.cudnn.benchmark = True

        self.num_classes = num_classes

        self.z_, self.y_ = prepare_z_y(self.batch_size, self.G.dim_z, num_classes,
                            device=self.device, fp16=config['fp16'])

        self.fixed_z, self.fixed_y = prepare_z_y(self.batch_size, self.G.dim_z,
                                    num_classes, device=self.device,
                                    fp16=config['fp16'])  
        self.fixed_z.sample_()
        self.fixed_y.sample_()

        # Inception for evaluation using pre-computed moments
        # Initialize inception network if evaluation is enabled
        self.enable_eval = config.get('enable_eval', False)
        if self.enable_eval:
            try:
                self.inception_net = inception_utils.load_inception_net(parallel=False)
                # Create sample function for inception evaluation
                def sample_fn():
                    """Sample function for inception evaluation."""
                    self.z_.sample_()
                    self.y_.sample_()
                    G_to_use = (self.G_ema if self.config['ema'] and self.config.get('use_ema', False)
                                else self.G)
                    images = G_to_use(self.z_, G_to_use.shared(self.y_))
                    return images, self.y_
                self.sample_for_inception = sample_fn
                
                # Initialize tracking variables
                self.best_IS = 0.0
                self.best_FID = 999999.0
                
                # Load pre-computed real moments from data folder
                import numpy as np
                dataset_name = config.get('eval_moments_file')
                
                # Try to find moments file in data folder or current directory
                moments_file = Path(dataset_name)
                try:
                    logger.info(f"Loading pre-computed inception moments from {moments_file}")
                    moments_data = np.load(moments_file)
                    self.real_mu = torch.tensor(moments_data['mu'], dtype=torch.float32).to(self.device)
                    self.real_sigma = torch.tensor(moments_data['sigma'], dtype=torch.float32).to(self.device)
                    logger.info("Successfully loaded pre-computed moments")
                except Exception as e:
                    logger.warning(f"Failed to load moments from {moments_file}: {e}")
                    self.enable_eval = False
            except Exception as e:
                logger.warning(f"Failed to initialize inception network for evaluation: {e}")
                self.enable_eval = False

    def forward(self, x, label= None):
        y = label.to(torch.long)
        flag = x.shape[0] != self.batch_size
        # if isinstance(batch, tuple) or isinstance(batch, list):
        #     x, y = batch  # real images and labels
        # else:
        #     x = batch
        #     y = args[0].to(torch.long)
        #     flag = True
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Discriminator step
        toggle_grad(self.D, True)
        toggle_grad(self.G, False)
        self.z_.sample_()
        self.y_.sample_()
        D_fake, D_real = self.GD(self.z_, self.y_, 
                            x, y, train_G=False, 
                            split_D=self.config['split_D'])

        D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake)

        toggle_grad(self.D, False)
        toggle_grad(self.G, True)
        # Generator step
        self.z_.sample_()
        self.y_.sample_()
        D_fake = self.GD(self.z_, self.y_, train_G=True, split_D=self.config['split_D'])
        G_loss = losses.generator_loss(D_fake)

        toggle_grad(self.D, True)
        toggle_grad(self.G, True)

        if self.config['ema']:
            self.ema.update()
        

        # Return losses in a tensor format compatible with distributed setup
        # D_loss at index 0, G_loss at index -1
        # Hack to serialize the losses correctly since the output needs to be a tensor with the same shape as the batch size and of type tensor, not distribution
        out = torch.zeros(self.batch_size, device=D_loss.device, dtype=D_loss.dtype)
        out[0] = D_loss
        out[-1] = G_loss

        if flag:
            return D_fake
        return out

    @torch.no_grad()
    def evaluate(self, step):
        """Evaluate model using Inception Score and FID."""
        if not self.enable_eval or self.inception_net is None:
            logger.debug("Evaluation disabled or inception network not initialized")
            return None
        
        # Check if pre-computed moments are available
        if (self.real_mu is None) or (self.real_sigma is None):
            logger.warning('Pre-computed real moments not available for FID; skipping evaluation.')
            return None
        
        # Accumulate generated activations
        pool, logits, labels = inception_utils.accumulate_inception_activations(
            self.sample_for_inception, self.inception_net, self.config.get('num_inception_images', 50000)
        )
        IS_mean, IS_std = inception_utils.calculate_inception_score(logits.cpu().numpy(), num_splits=10)
        mu_g = torch.mean(pool, 0)
        sigma_g = inception_utils.torch_cov(pool, rowvar=False)
        FID = inception_utils.torch_calculate_frechet_distance(mu_g, sigma_g, self.real_mu, self.real_sigma)
        FID = float(FID.cpu().numpy())
        
        logger.info(f'Step {step}: Inception Score is {IS_mean:.3f} +/- {IS_std:.3f}, FID is {FID:.4f}')
        
        self.best_IS = max(self.best_IS, IS_mean)
        self.best_FID = min(self.best_FID, FID)

        return {"IS_mean": IS_mean, "IS_std": IS_std, "FID": FID, "best_IS": self.best_IS, "best_FID": self.best_FID}