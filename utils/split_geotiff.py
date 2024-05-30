import datetime

from osgeo import gdal


def split_geotiff(input_file, output_folder, tile_size):
    dataset = gdal.Open(input_file)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    tile_geotransform = dataset.GetGeoTransform()
    tile_projection = dataset.GetProjection()

    for i in range(0, cols, tile_size):
        for j in range(0, rows, tile_size):
            tile_width = min(tile_size, cols - i)
            tile_height = min(tile_size, rows - j)

            output_tile_name = f"{output_folder}/tile_{i}_{j}.tif"
            driver = gdal.GetDriverByName("GTiff")
            output_tile = driver.Create(output_tile_name, tile_width, tile_height, dataset.RasterCount, gdal.GDT_Byte)

            list_tile_geotransform = list(tile_geotransform)
            list_tile_geotransform[0] += i * tile_geotransform[1]
            list_tile_geotransform[3] += j * tile_geotransform[5]

            output_tile.SetGeoTransform(list_tile_geotransform)
            output_tile.SetProjection(tile_projection)

            for band_num in range(1, dataset.RasterCount + 1):
                band = dataset.GetRasterBand(band_num)
                data = band.ReadAsArray(i, j, tile_width, tile_height)
                output_tile.GetRasterBand(band_num).WriteArray(data)

            output_tile = None

    dataset = None