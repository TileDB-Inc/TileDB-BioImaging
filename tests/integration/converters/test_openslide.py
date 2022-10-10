import openslide as osld

from tests import check_level_info, download_from_s3
from tiledbimg.openslide import TileDBOpenSlide


class TestOpenSlide:
    g_uri = "s3://tiledb-isaiah2/jjdemo/test4-convert/C3N-02572-22.tdg"
    svs_uri = "s3://tiledb-isaiah2/jjdemo/test4-convert/C3N-02572-22.svs"

    def test_openslide(self):
        t = TileDBOpenSlide.from_group_uri(self.g_uri)

        for l_num, l_info in enumerate(t.level_info):
            check_level_info(l_num, l_info)

        os_img = osld.open_slide(download_from_s3(self.svs_uri))
        assert t.dimensions == os_img.dimensions
        assert t.level_count == os_img.level_count
        assert t.level_dimensions == os_img.level_dimensions
        assert t.level_downsamples == os_img.level_downsamples
        for factor, best_level in (32, 2), (3.9, 0), (4.1, 1), (2.9, 0), (1, 0):
            assert os_img.get_best_level_for_downsample(factor) == best_level
            assert t.get_best_level_for_downsample(factor) == best_level
