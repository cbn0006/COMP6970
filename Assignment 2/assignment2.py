from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal
import cv2
from IPython.display import Image as ImageIPy
from IPython.display import display
import time
import ncc
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

class Assignment(object):

    '''
    Function:
    Create a Gaussian Pyramid from the input image.

    Parameters:
    Image - The input image to create Gaussian Pyramid.
    Scale - Factor by which every image is scaled down by.
    Minsize - Minimum size of the last level in the pyramid.

    Returns:
    Pyramid - A list of PIL Images.
    '''
    def MakeGaussianPyramid(self, image, scale, minsize):
        # Convert image to np array
        image_np = np.array(image)
        
        # List to return
        pyramid = [image]

        # Image width and height
        width, height = image.size

        # While greater than minsize, keep going
        while min(width, height) >= minsize:
            # Sigma for Gaussian Filter
            sigma = 1 / (2 * scale)
            # If RGB, do this
            if len(image_np.shape) == 3:
                filtered_image = np.zeros_like(image_np)
                # For each color channel, filter
                for i in range(3):
                    filtered_image[:, :, i] = gaussian_filter(image_np[:, :, i], sigma=sigma)
            # If greyscale, do this
            else:
                filtered_image = gaussian_filter(image_np, sigma=sigma)
            
            # Recalculate width and height
            width, height = int(width * scale), int(height * scale)
            if width < minsize or height < minsize:
                break

            # Resize image
            resized_image = Image.fromarray(filtered_image).resize((width, height), Image.BICUBIC)

            # Add image to pyramid
            pyramid.append(resized_image)

            # Initializing array for next image
            image_np = np.array(resized_image)

        return pyramid
    
    '''
    Function:
    Takes a PIL Image list as an input and uses imshow to display the pyramid in a joined, horizontal image.

    Parameters:
    Pyramid - A list of PIL Images.
    '''
    def ShowGaussianPyramid(self, pyramid):
        # Width and height
        total_width = sum(img.size[0] for img in pyramid)
        max_height = max(img.size[1] for img in pyramid)

        # Decided if final image is greyscale or RGB
        mode = pyramid[0].mode
        if mode == "L":
            final_image = Image.new("L", (total_width, max_height), color=255)
        else:
            final_image = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

        # For each image in pyramid, "connect" together
        x_offset = 0
        for img in pyramid:
            final_image.paste(img, (x_offset, 0))
            x_offset += img.size[0]

        # Convert final image to np array
        final_image_np = np.array(final_image)

        # Show Gaussian Pyramid
        plt.imshow(final_image_np, cmap="gray" if mode == "L" else None)
        plt.axis('off')
        plt.show()

    '''
    Function:
    Searches for template within the Gaussian Pyramid (pyramid) and creates a box around a template match in the original image.

    Parameters:
    Pyramid - A list of PIL Images
    Template - The image to search for in the pyramid.
    Threshold - The NCC threshold used to determine if there is a match or not.

    Returns:
    Image - The original image with red rectangles drawn around instances of the template.
    Count - Number of matches found between the original image and the template. 
    '''
    def FindTemplate(self, pyramid, template, threshold):
        # Detected matches
        count = 0

        # Fix the template width and resize based on that
        template_width = 10
        aspect_ratio = template.size[1] / template.size[0]
        template_resized = template.resize((template_width, int(template_width * aspect_ratio)), Image.BICUBIC)

        # Turn template gray and get dimensions
        template_gray = template_resized.convert('L')
        template_np = np.array(template_gray)
        h_t, w_t = template_np.shape

        # Make image RGB for drawing
        image = pyramid[0].copy().convert('RGB')
        draw = ImageDraw.Draw(image)

        # For each level in the pyramid, computer ncc and draw boxes if match is found
        for level, img in enumerate(pyramid):
            # Turn image gray and into np array
            img_gray = img.convert('L')
            img_np = np.array(img_gray)

            # Computer NCC
            ncc_result = ncc.normxcorr2D(img_np, template_np)

            # Get matches
            matches = np.where(ncc_result >= threshold)

            # Calculate scale factor
            scale_factor = math.pow(0.75, level)

            # For each match, draw a box on the orginal image
            for (y, x) in zip(*matches):
                x_adj = x - w_t // 2
                y_adj = y - h_t // 2

                x0 = int(x_adj / scale_factor)
                y0 = int(y_adj / scale_factor)
                x1 = int((x_adj + w_t) / scale_factor)
                y1 = int((y_adj + h_t) / scale_factor)

                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
                count += 1

        del draw

        return image, count
    
    '''
    Function:
    Creates a Laplacian Pyramid from the input image.

    Parameters:
    Image - The input image to create Laplacian Pyramid.
    Scale - Factor by which every image is scaled down by.
    Minsize - Minimum size of the last level in the pyramid.

    Returns:
    Laplacian Pyramid - A list of PIL Images.
    '''
    def MakeLaplacianPyramid(self, image, scale, minsize):
        # Create Gaussian Pyramid for image
        gaussian_pyramid = self.MakeGaussianPyramid(image, scale, minsize)
        # self.ShowGaussianPyramid(gaussian_pyramid)

        # Initialize pyramid list and get length variable
        laplacian_pyramid = []
        num_levels = len(gaussian_pyramid)

        # For each level + 1, recalculate previous level's Laplacian
        for i in range(num_levels - 1):
            # Current Gaussian image
            gauss_img = gaussian_pyramid[i]
            
            # Next Gaussian image
            gauss_img_next = gaussian_pyramid[i + 1]

            # Resize Gaussian image
            gauss_img_next_expanded = gauss_img_next.resize(gauss_img.size, Image.BICUBIC)

            # Convert images into np arrays for calculations
            gauss_img_np = np.array(gauss_img, dtype=np.float32)
            gauss_img_next_expanded_np = np.array(gauss_img_next_expanded, dtype=np.float32)

            # Compute Laplacian
            lap_img_np = gauss_img_np - gauss_img_next_expanded_np
            lap_img_np = np.clip(lap_img_np, 0, 255).astype(np.uint8)

            # Convert back to PIL
            lap_img = Image.fromarray(lap_img_np)

            # Append to Laplcian Pyramid
            laplacian_pyramid.append(lap_img)
        
        # Append last level
        gauss_img_last = gaussian_pyramid[-1]
        laplacian_pyramid.append(gauss_img_last)

        return laplacian_pyramid
    
    '''
    Function:
    Takes a PIL Image list as an input and uses imshow to display the pyramid in a joined, horizontal image.

    Parameters:
    Pyramid - A list of PIL Images.
    '''
    def ShowLaplacianPyramid(self, pyramid):
        # Initialize for adjusted images
        adjusted_images = []

        # For each level in pyramid, add them to the list
        for i, laplacian_level in enumerate(pyramid):
            laplacian_np = np.array(laplacian_level, dtype=np.float32)

            # If last image, calculate differently
            if i < len(pyramid) - 1:
                laplacian_display = np.clip(laplacian_np + 128, 0, 255).astype(np.uint8)
            else:
                laplacian_display = np.clip(laplacian_np, 0, 255).astype(np.uint8)

            # Convert back to np array
            laplacian_image = Image.fromarray(laplacian_display)
            adjusted_images.append(laplacian_image)

        # Total width and height
        total_width = sum(img.size[0] for img in adjusted_images)
        max_height = max(img.size[1] for img in adjusted_images)

        # Create an empty image for pasting
        if adjusted_images[0].mode == 'L':
            final_image = Image.new('L', (total_width, max_height), color=255)
        else:
            final_image = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))

        # Paste
        x_offset = 0
        for img in adjusted_images:
            final_image.paste(img, (x_offset, 0))
            x_offset += img.width

        # Display
        final_image_np = np.array(final_image)
        plt.imshow(final_image_np)
        plt.axis('off')
        plt.show()

    '''
    Function:
    Reconstruct Gaussian Pyramid with Laplacian Pyramid

    Parameters:
    lPyramid - Laplacian Pyramid that will be used to reconstruct the Gaussian Pyramid.

    Returns:
    gPyramid - The reconstructed Gaussian Pyramid.
    '''
    def ReconstructGaussianFromLaplacianPyramid(self, lPyramid):
        # Initialize empty list for Gaussian Pyramid and get number of levels
        gPyramid = []
        num_levels = len(lPyramid)
        
        # Initialize current level that will be worked on
        current_level = lPyramid[-1]

        # Last level is the same
        gPyramid.insert(0, current_level)

        # (Reverse order) For each level in the laplacian pyramid, reconstruct the correlated Gaussian pyramid level
        for i in range(num_levels - 2, -1, -1):
            # Current Laplacian level image
            laplacian_level = lPyramid[i]

            # Convert to np
            laplacian_np = np.array(laplacian_level, dtype=np.float32)

            # Resize the image
            current_image_resized = current_level.resize(laplacian_level.size, Image.BICUBIC)
            current_image_resized_np = np.array(current_image_resized, dtype=np.float32)

            # Reconstruct the image
            reconstructed_image_np = current_image_resized_np + laplacian_np
            reconstructed_image_np = np.clip(reconstructed_image_np, 0, 255).astype(np.uint8)

            # Convert to PIL
            reconstructed_image = Image.fromarray(reconstructed_image_np)

            # Insert to front of Gaussian Pyramid
            gPyramid.insert(0, reconstructed_image)

            # Update level
            current_level = reconstructed_image

        return gPyramid
    
    '''
    Function:
    Blends two images together using a mask and Laplacian Pyramids of each image.

    Parameters:
    imageA - The first image.
    imageB - The second image.
    mask - The mask used to blend the two images.
    '''
    def BlendImages(self, imageA, imageB, mask):
        # Turn mask into Gaussian Pyramid
        gaussian_pyramid_mask = assignment.MakeGaussianPyramid(mask, scale=0.75, minsize=16)

        # Get Laplacian Pyramids for both images
        laplacian_pyramid_A = assignment.MakeLaplacianPyramid(imageA, scale=0.75, minsize=16)
        laplacian_pyramid_B = assignment.MakeLaplacianPyramid(imageB, scale=0.75, minsize=16)

        # For each level in the mask, normalize
        normalized_mask_pyramid = []
        for mask_level in gaussian_pyramid_mask:
            mask_np = np.array(mask_level, dtype=np.float32) / 255.0
            if len(mask_np.shape) == 2:
                mask_np = np.stack([mask_np] * 3, axis=-1)
            normalized_mask_pyramid.append(mask_np)

        # For each level in the images, blend together with respect to the mask
        blended_pyramid = []
        for i in range(len(laplacian_pyramid_A)):
            lapA = np.array(laplacian_pyramid_A[i], dtype=np.float32)
            lapB = np.array(laplacian_pyramid_B[i], dtype=np.float32)
            mask = normalized_mask_pyramid[i]

            blended_lap = lapA * mask + lapB * (1 - mask)

            blended_lap = np.clip(blended_lap, 0, 255).astype(np.uint8)
            blended_image = Image.fromarray(blended_lap)

            blended_pyramid.append(blended_image)

        # Get image by reconstructing blended Laplacian Pyramid
        reconstructed_image = self.ReconstructGaussianFromLaplacianPyramid(blended_pyramid)

        # Display
        final_image = reconstructed_image[0]
        final_image.show()
        # final_image.save('orchid_violet_blend.jpg')

    '''
    Function:
    Compute the sum square difference between the texture image and a patch at each location in the texture image.

    Parameters:
    TODOPatch - A patch to be filled.
    TODOMask - A mask that determines useful and unuseful pixels.
    textureIM - Image being compared to patch.
    patchL - Number used to determine the patch size.

    Returns:
    SSD - An array containing the SSD between a patch and the image.
    '''
    def ComputeSSD(self, TODOPatch, TODOMask, textureIm, patchL):
        patch_rows, patch_cols, patch_bands = np.shape(TODOPatch)
        tex_rows, tex_cols, tex_bands = np.shape(textureIm)
        ssd_rows = tex_rows - 2 * patchL
        ssd_cols = tex_cols - 2 * patchL
        SSD = np.zeros((ssd_rows,ssd_cols))
        for r in range(ssd_rows):
            for c in range(ssd_cols):
                # Compute sum square difference between textureIm and TODOPatch
                # for all pixels where TODOMask = 0, and store the result in SSD
                # Get a patch at (r, c)
                texture_patch = textureIm[r:r + patch_rows, c:c + patch_cols, :]

                # Compute SSD between patch and image patch
                difference = (TODOPatch - texture_patch) ** 2

                # Apply mask to each color channel
                masked_difference = np.sum(difference * (1 - TODOMask[:, :, None]), axis=2)
                
                # Store result of SSD for patch at (r, c)
                SSD[r, c] = np.sum(masked_difference)
        return SSD
    
    '''
    Function:

    Parameters:
    imHole - Image that needs to be filled.
    TODOMask - A mask that determines useful and unuseful pixels.
    textureIm - unused.
    iPatchCenter - Center of patch to be filled (row).
    jPatchCenter - Center of patch to be filled (col).
    iMatchCenter - unused.
    jMatchCenter - unused.
    patchL - Number used to determine the patch size.
    '''
    def CopyPatch(self, imHole,TODOMask,textureIm,iPatchCenter,jPatchCenter,iMatchCenter,jMatchCenter,patchL):
        patchSize = 2 * patchL + 1
        for i in range(patchSize):
            for j in range(patchSize):
                # Copy the selected patch selectPatch into the image containing
                # the hole imHole for each pixel where TODOMask = 1.
                # The patch is centred on iPatchCenter, jPatchCenter in the image imHole
                if TODOMask[i, j] == 1:
                    # Position for patch
                    iHole = iPatchCenter - patchL + i
                    jHole = jPatchCenter - patchL + j
                    
                    # Paste patch to pixel in imHole
                    imHole[iHole, jHole, :] = selectPatch[i, j, :]
        return imHole
    
if __name__ == "__main__":
    assignment = Assignment()

    # 1.2 & 1.3
    template_image = Image.open('template.jpg')
    # pyramid = assignment.MakeGaussianPyramid(template_image, scale=0.75, minsize=16)
    # assignment.ShowGaussianPyramid(pyramid)
    # judybats = Image.open('judybats.jpg')
    # judyPyramid = assignment.MakeGaussianPyramid(judybats, scale=0.75, minsize=16)
    # assignment.ShowGaussianPyramid(judyPyramid)
    # family = Image.open('family.jpg')
    # familyPyramid = assignment.MakeGaussianPyramid(family, scale=0.75, minsize=16)
    # assignment.ShowGaussianPyramid(familyPyramid)
    # fans = Image.open('fans.jpg')
    # fansPyramid = assignment.MakeGaussianPyramid(fans, scale=0.75, minsize=16)
    # assignment.ShowGaussianPyramid(fansPyramid)

    # # 1.4
    # result_image, matches = assignment.FindTemplate(judyPyramid, template_image, threshold=0.7)
    # result_image.save('1.4.jpg')
    # print(f'Result image saved. {matches} matches were found.')

    # 1.5
    # image_filenames = ['judybats', 'students', 'tree', 'family', 'fans', 'sports']
    # ground_truth_faces = {
    #     'judybats.jpg': 3,
    #     'students.jpg': 27,
    #     'tree.jpg': 0,
    #     'family.jpg': 3,
    #     'fans.jpg': 3,
    #     'sports.jpg': 1
    # }
    # threshold = 0.62
    # results = []
    # for filename in image_filenames:
    #     input_image = Image.open(f'{filename}.jpg')
    #     pyramid = assignment.MakeGaussianPyramid(input_image, scale=0.75, minsize=16)
    #     result_image, count = assignment.FindTemplate(pyramid, template_image, threshold=threshold)
    #     result_image.save(f'1.5/result_{filename}_{threshold}.jpg')
    #     print(f'Result image saved to result_{filename}.jpg. {count} matches were found.')

    # 0.5 threshold way too low. (153, 522, 122, 114, 244, 19)
    # 0.6 threshold still too low. (37, 162, 5, 7, 39, 2)
    # 0.7 threshold too high. (10, 41, 0, 0, 0, 0)
    # 0.65 threshold seems best for error rate. (19, 84, 0, 1, 3, 0)
    # 0.62 threshold good. (28, 128, 2, 4, 19, 0)

    # 1.6: When it is actually yes, how often is it yes? Give a rate (need numbers)
    # The recall rate for each image with a 0.65 threshold is: 1, 0.67, no detection (Good), 0.33, 0, 0
    # The recall rate for each image with a 0.62 threshold is: 0.67, 0.74, 0, 0.67, 0, 0

    # 2.2
    orchid_image = Image.open('orchid.jpg')
    violet_image = Image.open('violet.jpg')
    laplacian_pyramid_orchid = assignment.MakeLaplacianPyramid(orchid_image, scale=0.75, minsize=16)
    laplacian_pyramid_violet = assignment.MakeLaplacianPyramid(violet_image, scale=0.75, minsize=16)

    # 2.3
    assignment.ShowLaplacianPyramid(laplacian_pyramid_orchid)
    assignment.ShowLaplacianPyramid(laplacian_pyramid_violet)

    # 2.4
    reconstructed_gaussian_pyramid = assignment.ReconstructGaussianFromLaplacianPyramid(laplacian_pyramid_violet)
    assignment.ShowGaussianPyramid(reconstructed_gaussian_pyramid)
    reconstructed_gaussian_pyramid = assignment.ReconstructGaussianFromLaplacianPyramid(laplacian_pyramid_orchid)
    assignment.ShowGaussianPyramid(reconstructed_gaussian_pyramid)

    # 2.5
    orchid_mask_image = Image.open('orchid_mask.bmp')
    orchid_mask_pyramid = assignment.MakeGaussianPyramid(orchid_mask_image, scale=0.75, minsize=16)
    # assignment.ShowGaussianPyramid(orchid_mask_pyramid)

    # 2.6
    imageA = Image.open('blue_cup.jpg')
    imageB = Image.open('green_cup.jpg')
    mask_image = Image.open('cup_mask.bmp')

    assignment.BlendImages(imageA, imageB, mask_image)

    imageA = Image.open('apple.jpg')
    imageB = Image.open('tomato.jpg')
    mask_image = Image.open('tomato_mask.bmp')

    assignment.BlendImages(imageA, imageB, mask_image)

    imageA = Image.open('orchid.jpg')
    imageB = Image.open('violet.jpg')
    mask_image = Image.open('orchid_mask.bmp')

    assignment.BlendImages(imageA, imageB, mask_image)