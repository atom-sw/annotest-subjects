from Tensorflow_progressive_growing_of_gans.model import Discriminator


def test_failing():
    num_channels = 1
    resolution = 32
    label_size = 0
    fmap_base = 4096
    fmap_decay = 1.0
    fmap_max = 256
    mbstat_func = 'Tstdeps'
    mbstat_avg = 'all'
    mbdisc_kernels = None
    use_wscale = True
    use_gdrop = True
    use_layernorm = False

    Discriminator(num_channels, resolution, label_size, fmap_base,
                  fmap_decay, fmap_max, mbstat_func, mbstat_avg, mbdisc_kernels,
                  use_wscale, use_gdrop, use_layernorm)
