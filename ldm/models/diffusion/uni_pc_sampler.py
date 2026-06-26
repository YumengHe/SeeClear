import torch
from .uni_pc import NoiseScheduleVP, model_wrapper, UniPC

class UniPCSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        self.alphas_cumprod = to_torch(model.alphas_cumprod)

    @torch.no_grad()
    def sample(self, S, batch_size, shape, conditioning=None, **kwargs):
        test_model_kwargs = kwargs.get('test_model_kwargs', {})
        if not test_model_kwargs:
            raise ValueError("UniPC sampler requires 'inpaint_image' and 'heatmap' (mask) inputs")
        
        z_inpaint = test_model_kwargs['inpaint_image']
        mask_resized = test_model_kwargs['heatmap']

        def unet_concat_wrapper(x, t, c):
            x_input = torch.cat([x, z_inpaint, mask_resized], dim=1)
            return self.model.apply_model(x_input, t, c)

        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)

        model_fn = model_wrapper(
            unet_concat_wrapper,
            ns,
            model_type="noise",
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=kwargs.get('unconditional_conditioning'),
            guidance_scale=kwargs.get('unconditional_guidance_scale', 1.0),
        )

        uni_pc = UniPC(model_fn, ns, algorithm_type="data_prediction", variant='bh2')

        device = self.model.device
        C, H, W = shape
        img = torch.randn((batch_size, C, H, W), device=device)
        
        x = uni_pc.sample(img, steps=S, skip_type="time_uniform", method="multistep", order=3, lower_order_final=True)

        return x, None