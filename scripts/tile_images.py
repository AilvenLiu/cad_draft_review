import os
from utils.image_tiling import tile_image

def main():
    png_dir = 'data/processed/pngs/'
    tiles_dir_base = 'data/processed/tiles/'
    if not os.path.exists(tiles_dir_base):
        os.makedirs(tiles_dir_base)
    
    for png_file in os.listdir(png_dir):
        if png_file.endswith('.png'):
            image_path = os.path.join(png_dir, png_file)
            tiles_subdir = os.path.join(tiles_dir_base, os.path.splitext(png_file)[0])
            tile_image(image_path, tiles_subdir)

if __name__ == "__main__":
    main()