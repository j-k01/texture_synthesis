import numpy as np
from scipy.ndimage import binary_dilation
import random
import matplotlib.pyplot as plt
import os
from PIL import Image

# ======== CONFIGURABLE PARAMETERS ========
SEED_IMAGE_PATH = '10_small.png'
OUTPUT_SQUARE_SIZE = 400  # Final image dimensions (square)
WINDOW_SIZE = 32  # Pattern matching window size
MAX_ERR_THRESHOLD = 0.075  # Error threshold for pattern matching
OUTPUT_DIR = 'output10_32'  # Directory for progress snapshots
SNAPSHOT_INTERVAL = 500  # Save image every X filled pixels
OUTPUT_FILENAME = 'synthesized_image_10_32.png'  # Final output filename

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_gaussian_window(window_size):
    x = np.arange(window_size) - window_size // 2
    y = np.arange(window_size) - window_size // 2
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-0.5 * (X ** 2 + Y ** 2) / (window_size / 6.4) ** 2)
    Z = Z / np.sum(Z)
    return np.stack((Z, Z, Z), axis=-1)  # Repeat the Gaussian window for all channels


def getUnfilledNeighbors(image):
    neighbors = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if np.any(image[i, j] == 0):  # Check if any channel is unfilled
                neighbors.append((i, j))
    return neighbors


def get_pixel_list(mask):
    struct = np.ones((3, 3), dtype=bool)
    dilated_mask = binary_dilation(mask == 1, structure=struct)

    # Edge mask: identifies the unfilled pixels adjacent to filled pixels
    edge_mask = dilated_mask & (mask == 0)

    # Get coordinates of potential pixels to fill
    valid_pixel_locations = np.where(edge_mask)
    valid_pixels = list(zip(valid_pixel_locations[0], valid_pixel_locations[1]))

    # Random permutation of valid_pixels
    random.shuffle(valid_pixels)
    valid_pixels = sorted(valid_pixels, key=lambda x: np.sum(mask[x[0] - 1:x[0] + 2, x[1] - 1:x[1] + 2]), reverse=True)

    return valid_pixels


def GetNeighborhoodWindow(image, pixel, window_size):
    x, y = pixel
    half_size = window_size // 2
    return image[x - half_size:x + half_size + 1, y - half_size:y + half_size + 1]


def getNeighborhoodWindowMask(mask, pixel, window_size):
    x, y = pixel
    half_size = window_size // 2
    return mask[x - half_size:x + half_size + 1, y - half_size:y + half_size + 1]


def get_all_square_gen(Sample, window_size):
    h, w, _ = Sample.shape
    half_window_size = window_size // 2

    # Find all the possible x, y coordinates for the center of the window
    coords = [(i, j) for i in range(half_window_size, h - half_window_size) for j in
              range(half_window_size, w - half_window_size)]

    # Randomly permute the list
    random.shuffle(coords)

    # Yield all samples
    for i, j in coords:
        yield Sample[i - half_window_size:i + half_window_size + 1, j - half_window_size:j + half_window_size + 1], (
        i, j)


def findMatches(Template, Template_mask, Sample, MaxErrThreshold):
    window_size = Template.shape[0]
    gaussMask = generate_gaussian_window(window_size)
    Template_mask = np.expand_dims(Template_mask, axis=-1)
    Template_mask = np.repeat(Template_mask, 3, axis=2)

    TotWeight = np.sum(gaussMask * Template_mask)
    distances = []
    patches = get_all_square_gen(Sample, window_size)
    for patch, (i, j) in patches:
        dist = (Template - patch) ** 2
        SSD = np.sum(dist * Template_mask * gaussMask, axis=(0, 1, 2)) / TotWeight
        distances.append((SSD, (i, j)))
    min_SSD = min(distances, key=lambda x: x[0])[0]
    return [x for x in distances if x[0] <= min_SSD * (1 + MaxErrThreshold)]


def GrowImage(SampleImage, Image, Mask, WindowSize, MaxErrThreshold, output_dir):
    half_window_size = (WindowSize // 2)
    fillable_region = Mask[half_window_size:-half_window_size, half_window_size:-half_window_size]
    filled_pixels = 0
    while not np.all(fillable_region):
        progress = False
        PixelList = get_pixel_list(Mask)
        for pixel in PixelList:
            if (
                    pixel[0] < half_window_size
                    or pixel[0] >= Mask.shape[0] - half_window_size
                    or pixel[1] < half_window_size
                    or pixel[1] >= Mask.shape[1] - half_window_size
            ):
                continue
            if np.all(Mask[pixel]):  # Skip already filled pixels
                continue
            Template = GetNeighborhoodWindow(Image, pixel, WindowSize)
            Template_mask = getNeighborhoodWindowMask(Mask, pixel, WindowSize)
            matches = findMatches(Template, Template_mask, SampleImage, MaxErrThreshold)
            if matches:
                best_match = random.choice(matches)
                new_pixel_val = SampleImage[best_match[1]]
                x, y = pixel
                Image[x, y] = new_pixel_val
                Mask[x, y] = 1
                progress = True

                # Save an image snapshot at specified interval
                if filled_pixels % SNAPSHOT_INTERVAL == 0:
                    plt.imsave(f'{output_dir}/snapshot_c_{filled_pixels}.png', Image)
                    print(f'Filled pixel: ({x}, {y}) with value: {new_pixel_val}')

                filled_pixels += 1

        if not progress:
            MaxErrThreshold *= 1.1
    return Image


def main():
    # Load seed image
    seed_image = np.array(Image.open(SEED_IMAGE_PATH).convert('RGB')) / 255.0

    # Make sure square_size is at least as large as the seed image
    square_size = max(OUTPUT_SQUARE_SIZE, seed_image.shape[0], seed_image.shape[1])

    # Initialize the mask and output image
    output_image_size = (square_size, square_size, 3)
    output_image = np.zeros(output_image_size)
    mask = np.zeros(output_image_size[:2])

    # Calculate the starting and ending indices for placing the seed image (centered)
    start_row = (square_size - seed_image.shape[0]) // 2
    end_row = start_row + seed_image.shape[0]
    start_col = (square_size - seed_image.shape[1]) // 2
    end_col = start_col + seed_image.shape[1]

    # Place the entire seed image in the center of the output image
    output_image[start_row:end_row, start_col:end_col] = seed_image
    mask[start_row:end_row, start_col:end_col] = 1

    # Run synthesis
    synthesized_image = GrowImage(seed_image, output_image, mask, WINDOW_SIZE, MAX_ERR_THRESHOLD, OUTPUT_DIR)

    # Convert from 0-1 to 0-255
    synthesized_image = (synthesized_image * 255).astype(np.uint8)

    # Save final image
    Image.fromarray(synthesized_image).save(OUTPUT_FILENAME)

    # Visual output
    plt.figure(figsize=(10, 10))
    plt.imshow(synthesized_image / 255.0)
    plt.axis('off')
    plt.title('Synthesized Image')
    plt.savefig(f'plt_{OUTPUT_FILENAME}')

    print(f'Texture synthesis complete. Result saved as {OUTPUT_FILENAME}')


if __name__ == "__main__":
    main()