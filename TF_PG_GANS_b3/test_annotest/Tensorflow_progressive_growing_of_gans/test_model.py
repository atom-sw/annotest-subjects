import unittest
import hypothesis as hy
import hypothesis.strategies as st
from Tensorflow_progressive_growing_of_gans.model import Discriminator, Generator


class Test_TopLevelFunctions(unittest.TestCase):

    @hy.given(num_channels=st.one_of(st.integers(min_value=1, max_value=
        1000), st.just(1)), resolution=st.one_of(st.sampled_from([4, 8, 16,
        32, 64, 128, 256, 512, 1024, 2048, 4096]), st.just(32)), label_size
        =st.one_of(st.integers(min_value=0, max_value=10), st.just(0)),
        fmap_base=st.one_of(st.sampled_from([4, 8, 16, 32, 64, 128, 256, 
        512, 1024, 2048, 4096]), st.just(4096)), fmap_decay=st.one_of(st.
        floats(min_value=0, max_value=1, allow_nan=None, allow_infinity=
        None, width=64, exclude_min=True, exclude_max=False), st.just(1.0)),
        fmap_max=st.one_of(st.sampled_from([4, 8, 16, 32, 64, 128, 256, 512,
        1024, 2048, 4096]), st.just(256)), latent_size=st.one_of(st.
        integers(min_value=1, max_value=1000), st.just(None)),
        normalize_latents=st.one_of(st.sampled_from([True, False]), st.just
        (True)), use_wscale=st.one_of(st.sampled_from([True, False]), st.
        just(True)), use_pixelnorm=st.one_of(st.sampled_from([True, False]),
        st.just(True)), use_leakyrelu=st.one_of(st.sampled_from([True,
        False]), st.just(True)), use_batchnorm=st.one_of(st.sampled_from([
        True, False]), st.just(False)), tanh_at_end=st.one_of(st.floats(
        min_value=0, max_value=2.0, allow_nan=None, allow_infinity=None,
        width=64, exclude_min=False, exclude_max=False), st.just(None)))
    @hy.settings(deadline=None)
    def test_Generator(self, num_channels, resolution, label_size,
        fmap_base, fmap_decay, fmap_max, latent_size, normalize_latents,
        use_wscale, use_pixelnorm, use_leakyrelu, use_batchnorm, tanh_at_end):
        Generator(num_channels, resolution, label_size, fmap_base,
            fmap_decay, fmap_max, latent_size, normalize_latents,
            use_wscale, use_pixelnorm, use_leakyrelu, use_batchnorm,
            tanh_at_end)

    @hy.given(num_channels=st.just(1), resolution=st.just(32), label_size=
        st.just(0), fmap_base=st.just(4096), fmap_decay=st.just(1.0),
        fmap_max=st.just(256), mbstat_func=st.just('Tstdeps'), mbstat_avg=
        st.just('all'), mbdisc_kernels=st.just(None), use_wscale=st.just(
        True), use_gdrop=st.just(True), use_layernorm=st.just(False))
    @hy.settings(deadline=None)
    def test_Discriminator(self, num_channels, resolution, label_size,
        fmap_base, fmap_decay, fmap_max, mbstat_func, mbstat_avg,
        mbdisc_kernels, use_wscale, use_gdrop, use_layernorm):
        Discriminator(num_channels, resolution, label_size, fmap_base,
            fmap_decay, fmap_max, mbstat_func, mbstat_avg, mbdisc_kernels,
            use_wscale, use_gdrop, use_layernorm)
