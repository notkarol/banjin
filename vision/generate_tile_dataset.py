#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage
from PIL import Image, ImageDraw, ImageFont
import time
import os
import pickle
import seaborn
import argparse

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='train', help="base name of output file")
    parser.add_argument("--n_images", default=int(1E5), help="number of images in output file", type=int)
    parser.add_argument("--n_border", default=3, help="pixels of border around each letter", type=int)
    parser.add_argument("--n_fg_pixels", default=24, help="output image size", type=int)
    parser.add_argument("--n_bg_pixels", default=48, help="size of background for smooth rotates and warps", type=int)
    parser.add_argument("--min_font_size", default=16, help="smallest_font to draw", type=int)
    parser.add_argument("--max_font_size", default=23, help="largest font to draw", type=int)
    parser.add_argument("--p_letter", default=0.9, help="probability that we have a letter tile", type=float)
    parser.add_argument('--glyphs', default=' ABCDEFGHIJKLMNOPQRSTUVWXYZ', help='glyphs to generate for dataset')
    parser.add_argument('--fonts_folder', default='fonts', help='folder we can find font ttfs in')
    parser.add_argument('--plot', default=False, action='store_true', help='whether to plot')
    return parser.parse_args()

def plot_class_distribution(args, labels):
    # Show that we're generating more blank tiles.
    b = np.bincount(labels)
    plt.plot(b, '*-')
    plt.axis([-1, len(b), 0, np.max(b)])
    plt.title('Class Distribution')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.savefig('plots/%s-class-distribution.png' % args.name, dpi=160, bbox_inches='tight')

def plot_tile_positions(args, Y, n_offset, d_off_x, d_off_y):
    count_blank = np.zeros((n_offset * 2 + 1, n_offset * 2 + 1), dtype=np.float)
    count_letter = np.zeros((n_offset * 2 + 1, n_offset * 2 + 1), dtype=np.float)
    x = np.linspace(-n_offset, n_offset, n_offset * 2 + 1)
    y = np.linspace(-n_offset, n_offset, n_offset * 2 + 1)
    xv, yv = np.meshgrid(x, y)
    for i in range(len(Y)):
        if Y[i, 0] == 0:
            count_blank[d_off_y[i] + n_offset, d_off_x[i] + n_offset] += 1
        else:
            count_letter[d_off_y[i] + n_offset, d_off_x[i] + n_offset] += 1

    fig = plt.figure(figsize=(8,8))
    plt.scatter(xv, yv, s=count_blank, c='r', alpha=0.5)
    plt.scatter(xv, yv, s=count_letter, c='b', alpha=0.3)
    plt.title('Frequency of Tile Positions: blue is non-zero label, red is classified as none')
    plt.xlabel('X Offset')
    plt.ylabel('Y Offset')
    plt.axis('equal')
    plt.savefig('plots/%s-tile-positions.png' % args.name, dpi=160, bbox_inches='tight')

def plot_other_distributions(args, n_offset, d_angle, d_warp, d_font_size, range_font_sizes,
                             d_tile_noise_weight, d_font_noise_weight, d_blur_weight):
    fig, axs = plt.subplots(2, 2) ; fig.set_size_inches(10, 10)

    axs[0][0].hist(d_angle, bins=36)
    axs[0][0].set_xlabel("Angle")
    axs[0][0].set_ylabel("Count")
    axs[0][0].set_title('Rotation Angle')
    axs[0][0].set_xlim(0, 360)

    axs[0][1].set_title('Example Affine Transforms')
    for i in range(32):
        xs = [d_warp[i, 0, 0], d_warp[i, 1, 0], d_warp[i, 3, 0], d_warp[i, 2, 0], d_warp[i, 0, 0]]
        ys = [d_warp[i, 0, 1], d_warp[i, 1, 1], d_warp[i, 3, 0], d_warp[i, 2, 1], d_warp[i, 0, 1]]
        axs[0][1].plot(xs, ys)

    axs[0][1].set_xlim(0 - n_offset, args.n_bg_pixels + n_offset)
    axs[0][1].set_ylim(0 - n_offset, args.n_bg_pixels + n_offset)
    axs[0][1].set_xticks([0, args.n_bg_pixels])
    axs[0][1].set_yticks([0, args.n_bg_pixels])

    axs[1][0].hist(d_font_size, bins=len(range_font_sizes))
    axs[1][0].set_title('Font Sizes')

    axs[1][1].plot(np.sort(d_tile_noise_weight), label='Tile Noise')
    axs[1][1].plot(np.sort(d_font_noise_weight), label='Font Noise')
    axs[1][1].plot(np.sort(d_blur_weight), label='Blur')
    axs[1][1].set_title('Weights')
    axs[1][1].set_ylim(0, 1)
    axs[1][1].legend(loc=2)
    plt.savefig('plots/%s-distributions.png' % args.name, dpi=160, bbox_inches='tight')

def plot_fonts(args, range_font_sizes, font_filenames, imgs):
    # View some cutouts of how well the network did
    border = 1
    sx = (np.ptp(range_font_sizes) + 1) * len(font_filenames)
    sy = len(args.glyphs)

    imgs = np.array(imgs, dtype=np.uint8)

    # Prepare cutouts
    cutouts = np.zeros(((imgs.shape[1] + border) * sy, (imgs.shape[2] + border) * sx), dtype=np.uint8) + 128
    for i in range(min(sx * sy, len(imgs))):
        y = i // sx * (imgs.shape[1] + border)
        x = i  % sx * (imgs.shape[2] + border)
        cutouts[y : y + imgs.shape[1], x : x + imgs.shape[2]] = imgs[i, :, :]
    fig = plt.figure(figsize=(sx, sy))
    plt.imshow(cutouts, cmap='gray')
    plt.axis('off')
    plt.savefig('plots/%s-fonts.png' % args.name, dpi=160, bbox_inches='tight')

def plot_generation(args, n, bg, X, Y, d_letters):
    m = 7
    fig, axs = plt.subplots(n, m) ; fig.set_size_inches(m, n)
    for i in range(n):
        for j in range(m):
            axs[i][j].grid(False)
            axs[i][j].set_xticklabels([])
            axs[i][j].set_yticklabels([])
    axs[0][0].set_title('Background')
    axs[0][1].set_title('Tile')
    axs[0][2].set_title('Letter+Noise')
    axs[0][3].set_title('Rotate')
    axs[0][4].set_title('Affine')
    axs[0][5].set_title('Blur')
    axs[0][6].set_title('Crop')

    for i in range(n):
        axs[i][0].set_ylabel('%c %.2f' % (d_letters[i], Y[i, 1]))
        for j in range(m - 1):
            axs[i][j].imshow(bg[i, j], interpolation="nearest", vmin=0, vmax=255, cmap=plt.get_cmap('gray'))
        axs[i][m - 1].imshow(X[i, 0], interpolation="nearest", vmin=0, vmax=255, cmap=plt.get_cmap('gray'))

    plt.savefig('plots/%s-generation.png' % args.name, dpi=160, bbox_inches='tight')
    
def main():
    args = arguments()

    ## Common variables derived from arguments
    range_font_sizes = np.arange(args.min_font_size, args.max_font_size)
    corners = np.float32([[0, 0], [args.n_bg_pixels, 0], [0, args.n_bg_pixels], [args.n_bg_pixels, args.n_bg_pixels]])
    font_filenames = os.listdir(args.fonts_folder)
    n_offset = (args.n_bg_pixels - args.n_fg_pixels) // 2

    # Dataset X and Y
    X = np.zeros((args.n_images, 1, args.n_fg_pixels, args.n_fg_pixels), dtype=np.uint8)
    Y = np.zeros((args.n_images, 2), dtype=np.float32)

    # Prepare Y by filling the one-hot class in the first len(letter) columns and then the normalized distance in the last column
    labels = (np.random.randint(1, len(args.glyphs), size=args.n_images) *
              (np.random.rand(args.n_images) < args.p_letter))
    for i, label in enumerate(labels):
        Y[i, 0] = label

    ## Prepare background
    # Generate the shapes for the background
    d_bg_median_blur = np.random.choice(np.arange(3, args.n_bg_pixels // 4 + 1, 2), size=args.n_images)

    # Whether to show a tile or not. Always true for tiles, True half the time for blank tiles
    d_show_tile = np.logical_or(np.random.rand(args.n_images) < 0.5, Y[:, 0] > 0)

    # The size of the shadow
    d_shadow_x = np.random.randint(-args.n_fg_pixels // 16, args.n_fg_pixels // 16 + 1, size=args.n_images)
    d_shadow_y = np.random.randint(-args.n_fg_pixels // 16, args.n_fg_pixels // 16 + 1, size=args.n_images)
    d_shadow_weight = np.clip(np.random.normal(loc=0.0, scale=0.05, size=args.n_images), 1E-6, 1.0)

    # Distributions for while font file and font size to use
    d_font_file = np.random.randint(len(font_filenames), size=args.n_images)
    d_font_size = np.clip(np.array(np.random.normal(loc=np.mean(range_font_sizes), scale=np.log(np.ptp(range_font_sizes)), size=args.n_images) + 0.5, dtype=np.int), np.min(range_font_sizes), np.max(range_font_sizes))

    # The size of the blur affects the shapes generated by bluring
    d_tile_median_blur = np.random.choice(np.array(np.arange(3, args.n_fg_pixels // 4 + 1, 2), dtype=np.int), size=args.n_images)
    d_tile_noise_weight = np.clip(np.random.normal(loc=0.05, scale=0.05, size=args.n_images), 0.0, 0.5)
    d_font_noise_weight = np.clip(np.random.normal(loc=0.2, scale=0.1, size=args.n_images), 0.0, 0.5)

    # The directions in which to warp the tile. Ideally this would only be the warps that a
    # camera circling a tile from above would generate. Also serves as a form of noise and bluring.
    d_warp = np.float32(np.random.normal(loc=0, scale=args.n_fg_pixels / 12., size=(args.n_images, 4, 2))) + corners

    # Angle to rotate tile by
    d_angle = (np.random.normal(loc=0, scale=30, size=args.n_images) + np.random.randint(9, size=args.n_images) * 90) % 360

    # Ratio of bluring
    d_blur_weight = np.clip(np.random.normal(loc=0.05, scale=0.1, size=args.n_images), 0.0, 1.0)

    # Handle offsets. In the case of face-up tiles, the offset should be such
    # that smaller tiles have more room to move around
    d_off_x = np.clip(np.array(np.random.normal(loc=args.n_bg_pixels, scale=args.n_fg_pixels / 8., size=args.n_images) + 0.5, dtype=np.int) - args.n_bg_pixels, -(n_offset // 2), n_offset // 2)
    d_off_y = np.clip(np.array(np.random.normal(loc=args.n_bg_pixels, scale=args.n_fg_pixels / 8., size=args.n_images) + 0.5, dtype=np.int) - args.n_bg_pixels, -(n_offset // 2), n_offset // 2)

    # The last row of our Y labels is the distance by which we offset by
    # Should probably figure out a heuristic to tie in warps
    y = d_off_y - (np.mean(d_warp[:, :, 1] - corners[:,1], axis=1) // 4)
    x = d_off_x - (np.mean(d_warp[:, :, 0] - corners[:,0], axis=1) // 4)
    Y[:, 1] = np.sqrt(x * x + y * y)
    Y[:, 1] /= np.max(Y[Y[:, 0] > 0, -1])
    Y[:, 1] = np.clip(Y[:, -1], 0, 1.0)
    Y[Y[:, 0] == 0, 1] = 1.0

    n_half_offset = n_offset // 2 + 2
    for i in range(args.n_images):
        if Y[i, 0] == 0:
            off = np.array(np.random.rand(2) * args.n_fg_pixels + 0.5, dtype=np.int) - (args.n_fg_pixels // 2)
            while (-n_half_offset <= off[0] <= n_half_offset) and (-n_half_offset <= off[1] <= n_half_offset):
                off = np.array(np.random.rand(2) * args.n_fg_pixels + 0.5, dtype=np.int) - (args.n_fg_pixels // 2)
            d_off_x[i] = off[0]
            d_off_y[i] = off[1]

    # Get the char of the category we wish to place. In the case of the blank tile, 
    # occasionally make it a letter to show very offset tiles
    d_letters = [''] * args.n_images
    for i in range(args.n_images):
        pos = (Y[i, 0] if Y[i, 0] < 27 else
               np.random.randint(1, len(args.glyphs)) * (np.random.rand() < args.p_letter))
        d_letters[i] = args.glyphs[int(pos)]
            
    # When PIL draws tiles of different font sizes, they unfortunately are not centered.
    # This code stores the amount of offsets we need to center a tile and stores a font object 
    # To limit the number of file system hits
    rolls = {}
    fonts = {}
    imgs = []
    max_brightness_of_letter = 200
    for letter in args.glyphs:
        rolls[letter] = {}
        fonts[letter] = {}
        for font_filename in font_filenames:
            rolls[letter][font_filename] = {}
            fonts[letter][font_filename] = {}
            for font_size in range_font_sizes:
                fonts[letter][font_filename][font_size] = ImageFont.truetype(font_filename, font_size, encoding='unic')

                # Draw the letter 
                img = Image.new('L', (args.n_fg_pixels, args.n_fg_pixels), 'white')
                ImageDraw.Draw(img).text((0, 0), text=letter, font=fonts[letter][font_filename][font_size])
                img = np.array(img, dtype=np.uint8)

                # Find the number of whitespace on each side of the image
                for u in range(img.shape[0]):
                    if np.min(img[u, :]) < max_brightness_of_letter:
                        break
                for d in range(img.shape[0]):
                    if np.min(img[img.shape[0] - d - 1, :]) < max_brightness_of_letter:
                        break
                for l in range(img.shape[1]):
                    if np.min(img[:, l]) < max_brightness_of_letter:
                        break
                for r in range(img.shape[1]):
                    if np.min(img[:, img.shape[1] - r - 1]) < max_brightness_of_letter:
                        break
                # Record the offset required to center the letter on the tile
                rolls[letter][font_filename][font_size] = np.array(((r - l + 0.5) // 2, (d - u + 0.5) // 2), dtype=np.int)

                img = np.roll(img, rolls[letter][font_filename][font_size][0], axis=1)
                img = np.roll(img, rolls[letter][font_filename][font_size][1], axis=0)
                imgs.append(img)

        
    # Go through all our images
    n = 10
    bg = np.zeros((n, 7, args.n_bg_pixels, args.n_bg_pixels), dtype=np.float)
    for i in range(args.n_images):
        bg_i = 0
        
        # Plot how long it took to generate 1000 images
        if i % 100 == 99:
            print("%.2f%%" % (100 * (i + 1) /  args.n_images), end='\r')

        # Prepare background and renormalize
        bg[i % n, 0, :, :] = np.random.rand(args.n_bg_pixels, args.n_bg_pixels) * 255
        bg[i % n, 0, :, :] = cv2.medianBlur(np.array(bg[i % n, 0], dtype=np.uint8), d_bg_median_blur[i])
        bg[i % n, 0, :, :] -= np.min(bg[i % n, 0])
        bg[i % n, 0, :, :] *= 255. / np.max(bg[i % n, 0])
        mean_color = np.mean(bg[i % n, 0])


        # Place tile down for every letter, and randomly for non-glyphs. Add shadow
        bg[i % n, 1, :, :] = bg[i % n, 0]
        if d_show_tile[i]:
            off = n_offset + (args.n_fg_pixels - d_font_size[i]) // 2
            b = off - args.n_border
            a = off + args.n_border + d_font_size[i]

            # Create Shadow and round corners
            bg[i % n, 1, b + d_shadow_y[i] : a + d_shadow_y[i],
               b + d_shadow_x[i] : a + d_shadow_x[i]] *= d_shadow_weight[i]
            for y in [b + d_shadow_y[i], a + d_shadow_y[i] - 1]:
                for x in [b + d_shadow_x[i], a + d_shadow_x[i] - 1]:
                    bg[i % n, 1, y, x] /= d_shadow_weight[i]

            # Create Tile
            bg[i % n, 1, b : a, b : a] = 255 - bg[i % n, 1, b : a, b : a]

            # Round corners
            bg[i % n, 1, b, b] = 127
            bg[i % n, 1, b, a - 1] = 127
            bg[i % n, 1, a - 1, b] = 127
            bg[i % n, 1, a - 1, a - 1] = 127

        # Write letter onto background:
        img = Image.fromarray(bg[i % n, 1])
        roll = rolls[d_letters[i]][font_filenames[d_font_file[i]]][d_font_size[i]]
        font = fonts[d_letters[i]][font_filenames[d_font_file[i]]][d_font_size[i]]
        ImageDraw.Draw(img).text(roll + n_offset, text=d_letters[i], font=font, fill=None)
        bg[i % n, 2, :, :] = np.minimum(np.array(img, dtype=np.float), bg[i % n, 1])

        # Create chunky noise to add to tile
        if d_show_tile[i]:
            tile = bg[i % n, 2, b : a, b : a]
            noise = np.random.randint(256, size=tile.shape)
            noise[:, :] = cv2.medianBlur(np.array(noise, dtype=np.uint8), d_tile_median_blur[i])
            noise[:, :] = (noise - np.min(noise)) * (255. / (np.max(noise) - np.min(noise)))
                         
            # Apply noise to tile
            tile[tile < 64] = tile[tile < 64] * (1 - d_font_noise_weight[i]) + noise[tile < 64] * d_font_noise_weight[i]
            tile[:, :] =  tile * (1 - d_tile_noise_weight[i]) + noise * d_tile_noise_weight[i]
            

        # Rotate the letter
        bg[i % n, 3, :, :] = scipy.ndimage.rotate(bg[i % n, 2], d_angle[i], reshape=False, cval=mean_color)

        # Affine transform 
        transform = cv2.getPerspectiveTransform(corners, d_warp[i])
        bg[i % n, 4, :, :] = cv2.warpPerspective(bg[i % n, 3], transform, (args.n_bg_pixels, args.n_bg_pixels))

        # Blur
        bg[i % n, 5, :, :] = cv2.GaussianBlur(bg[i % n, 4], (5,5), 0) * d_blur_weight[i] + bg[i % n, 4] * (1 - d_blur_weight[i])

        # Copy over background
        y = n_offset + d_off_y[i]
        x = n_offset + d_off_x[i]
        X[i, 0, :, :] = np.clip(bg[i % n, 5, y : y + args.n_fg_pixels, x : x + args.n_fg_pixels], 0, 255)
    print()
    
    # make sure data directory exists
    if not os.path.isdir('data'):
        os.mkdir('data')
        
    # Save tiles to pickle
    coefficient = args.n_images // np.power(10, int(np.log10(args.n_images)))
    exponent = np.log10(args.n_images)
    filename = '%s_%iE%i' % (args.name, coefficient, exponent)
    with open(os.path.join('data', filename + '.pkl'), 'wb') as f:
        pickle.dump((X, Y), f)
    print('Saved', filename)

    # Plot the outcomes of this model
    if args.plot:
        if not os.path.isdir('plot'):
            os.mkdir('plot')
        plot_class_distribution(args, labels)
        plot_fonts(args, range_font_sizes, font_filenames, imgs)
        plot_tile_positions(args, Y, n_offset, d_off_x, d_off_y)
        plot_other_distributions(args, n_offset, d_angle, d_warp, d_font_size, range_font_sizes,
                                 d_tile_noise_weight, d_font_noise_weight, d_blur_weight)
        plot_generation(args, n, bg, X[-n:, 0, :, :], Y[-n:], d_letters[-10:])

# Run program
if __name__ == "__main__":
    main()
