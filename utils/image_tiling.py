from PIL import Image
import os

def tile_image(image_path, output_dir, tile_size=1024, overlap=100):
    """
    Splits a large image into smaller tiles with overlap.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the image tiles.
        tile_size (int): Size of each tile (tile_size x tile_size).
        overlap (int): Overlapping pixels between tiles.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image = Image.open(image_path)
    width, height = image.size
    x_steps = range(0, width, tile_size - overlap)
    y_steps = range(0, height, tile_size - overlap)
    
    for i, x in enumerate(x_steps):
        for j, y in enumerate(y_steps):
            box = (x, y, x + tile_size, y + tile_size)
            tile = image.crop(box)
            tile_filename = f"tile_{i}_{j}.png"
            tile.save(os.path.join(output_dir, tile_filename))
            print(f"Saved {os.path.join(output_dir, tile_filename)}")

# Example usage
if __name__ == "__main__":
    image_path = 'data/processed/pngs/page_1.png'
    tiles_dir = 'data/processed/tiles/page_1/'
    tile_image(image_path, tiles_dir)