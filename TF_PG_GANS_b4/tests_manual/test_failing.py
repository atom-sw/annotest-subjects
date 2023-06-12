from Tensorflow_progressive_growing_of_gans.model import Generator


def test_failing():
    num_channels = 1
    resolution = 32
    label_size = 0
    fmap_base = 4096
    fmap_decay = 1.0
    fmap_max = 256
    latent_size = None
    normalize_latents = True
    use_wscale = True
    use_pixelnorm = True
    use_leakyrelu = True
    use_batchnorm = False
    tanh_at_end = None

    Generator(num_channels, resolution, label_size, fmap_base,
              fmap_decay, fmap_max, latent_size, normalize_latents,
              use_wscale, use_pixelnorm, use_leakyrelu, use_batchnorm,
              tanh_at_end)
