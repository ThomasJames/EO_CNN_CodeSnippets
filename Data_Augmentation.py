
import math
from matplotlib.image import imsave
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

# Split the Images
def split_image(dim_pix, im, location, im_or_mask, folder, number):
    # Find the number of sub-Images that fit in rows
    rows = []
    for i in range((math.floor(im.shape[0] / dim_pix))):
        rows.append(i)
    # Find the number of sub-Images that fit in rows
    columns = []
    for i in range((math.floor(im.shape[1] / dim_pix))):
        columns.append(i)

    # Numerically identify the sub-Images
    a = 0
    for i in rows:
        for j in columns:
            # Check for 244 x 244 (Mask) or 244 x 244 x 3 (TC Images)
            if (im[0 + (dim_pix * j): dim_pix + (dim_pix * j),
                  0 + dim_pix * i: dim_pix + (dim_pix * i)].shape[0]) == dim_pix:
                if (im[0 + (dim_pix * j): dim_pix + (dim_pix * j),
                  0 + dim_pix * i: dim_pix + (dim_pix * i)].shape[1]) == dim_pix:

                    tile = im[0 + (dim_pix * j): dim_pix + (dim_pix * j),
                            0 + dim_pix * i: dim_pix + (dim_pix * i)]


                    # Stop white tiles for positive results
                    count = np.count_nonzero(tile == 1) == (dim_pix * dim_pix)
                    if count:
                        print(f"Tile {a} is only land")
                        all_black = np.tile(1, (dim_pix, dim_pix))
                        all_black[0][0] = 0
                        imsave(f"{folder}/{location}_{number}_{a}_{im_or_mask}.png",
                               all_black,
                               format="png",
                               cmap='Greys')
                    else:
                        # Save the 244 x 244 as an png file.
                        imsave(f"{folder}/{location}_{number}_{a}_{im_or_mask}.png",
                                tile,
                                format="png",
                                cmap='Greys')
                    a += 1
                else:
                    print("Out of shape")


# Salt and pepper
# Function by: Md. Rezwanul Haque (stolen from stack overflow)
def sp_noise(image, prob):
    '''
    Add salt and pepper noise to Images
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output



if __name__ == "__main__":

    """
    TC - Raw True colour Images
    GT - Ground Truth
    """
    for i in range(4):
        try:

            location = "Rotterdam"
            region = i

            print(f"{location}:{region}")
            # Load the raw Images
            TC = cv2.imread(f"Data/{location}_{region}/{location}_TC.png")
            TC = np.array(TC)

            print(TC.shape)

            plt.imshow(TC)
            plt.show()


            # Create a mask using the ground truth.
            # Convert the ground truth into a mask
            GT = cv2.imread(f"Data/{location}_{region}/{location}_GT.png")
            GT = np.array(GT)

            # Select one channel
            GT = GT[:, :, -1]

            # Generate a binary mask
            GT[GT > 0] = 1
            plt.imshow(GT, cmap="Blues")
            plt.show()


            TC_RAW = TC

            # plt.imsave(f"/Mask_Plots/GT_{region}_{location}.png", GT, cmap="tab20c")

            plt.imsave(f"/Users/tj/PycharmProjects/Semantic-Segmentation_DeepLabV3/Water_Mask_Eval/Mask_plots/{location}_{region}.png",
                       GT, cmap='tab20c')

            TC_noise = sp_noise(TC, 0.05)
            TC_Hflip = np.flip(TC, 1)
            GT_Hflip = np.flip(GT, 1)
            TC_Vflip = np.flip(TC, 0)
            GT_Vflip = np.flip(GT, 0)
            TC_Hflip_Vflip = np.flip(TC_Hflip, 0)
            GT_Hflip_Vflip = np.flip(GT_Hflip, 0)
            TC_Blur = cv2.medianBlur(TC, 5)
            TC_Blur_vflip = np.flip(TC_Blur, 0)
            TC_Noise_Vflip = np.flip(TC_noise, 0)
            TC_lab = lab= cv2.cvtColor(TC, cv2.COLOR_BGR2LAB)
            TC_enhanced = cv2.addWeighted(TC, 10, TC, 0, 10)

            plt.imsave("augplots/TC.png", TC)
            plt.imsave("augplots/TC_noise.png", TC_noise)
            plt.imsave("augplots/TC_Vflip.png", TC_Vflip)
            plt.imsave("augplots/TC_Hflip_Vflip.png", TC_Hflip_Vflip)
            plt.imsave("augplots/TC_Blur_vflip.png", TC_Blur_vflip)
            plt.imsave("augplots/TC_Noise_Vflip.png", TC_Noise_Vflip)
            plt.imsave("augplots/TC_lab.png", TC_lab)

            images_path = "/Users/tj/PycharmProjects/Semantic-Segmentation_DeepLabV3/DeepLabV3-Urban_Water_detection/Data/further_aug/Images"
            masks_path = "/Users/tj/PycharmProjects/Semantic-Segmentation_DeepLabV3/DeepLabV3-Urban_Water_detection/Data/further_aug/Masks"


            split_image(dim_pix=244, im=TC_RAW, location=location, im_or_mask=f"TC_RAW", number=f"Region_{region}",
                        folder=images_path)
            split_image(dim_pix=244, im=GT, location=location, im_or_mask=f"Mask_RAW", number=f"Region_{region}",
                        folder=masks_path)

            split_image(dim_pix=244, im=TC_noise, location=location, im_or_mask=f"TC_noise", number=f"Region_{region}",
                        folder=images_path)
            split_image(dim_pix=244, im=GT, location=location, im_or_mask=f"Mask_noise", number=f"Region_{region}",
                        folder=masks_path)


            split_image(dim_pix=244, im=TC_Hflip_Vflip, location=location, im_or_mask=f"TC_Hflip_Vflip", number=f"Region_{region}",
                        folder=images_path)
            split_image(dim_pix=244, im=GT_Hflip_Vflip, location=location, im_or_mask=f"Mask_Hflip_Vflip", number=f"Region_{region}",
                        folder=masks_path)

            split_image(dim_pix=244, im=TC_Hflip, location=location, im_or_mask=f"TC_Hflip", number=f"Region_{region}",
                        folder=images_path)
            split_image(dim_pix=244, im=GT_Hflip, location=location, im_or_mask=f"Mask_Hflip", number=f"Region_{region}",
                        folder=masks_path)


            split_image(dim_pix=244, im=TC_Vflip, location=location, im_or_mask=f"TC_Vflip", number=f"Region_{region}",
                        folder=images_path)
            split_image(dim_pix=244, im=GT_Vflip, location=location, im_or_mask=f"Mask_Vflip", number=f"Region_{region}",
                        folder=masks_path)


            split_image(dim_pix=244, im=TC_Blur, location=location, im_or_mask=f"TC_blur", number=f"Region_{region}",
                        folder=images_path)
            split_image(dim_pix=244, im=GT, location=location, im_or_mask=f"Mask_blur", number=f"Region_{region}",
                        folder=masks_path)


            split_image(dim_pix=244, im=TC_Blur_vflip, location=location, im_or_mask=f"TC_Blur_vflip", number=f"Region_{region}",
                        folder=images_path)
            split_image(dim_pix=244, im=GT_Vflip, location=location, im_or_mask=f"Mask_Blur_vflip", number=f"Region_{region}",
                        folder=masks_path)


            split_image(dim_pix=244, im=TC_Noise_Vflip, location=location, im_or_mask=f"TC_Noise_Vflip", number=f"Region_{region}",
                        folder=images_path)
            split_image(dim_pix=244, im=GT_Vflip, location=location, im_or_mask=f"Mask_Noise_Vflip", number=f"Region_{region}",
                        folder=masks_path)

            # split_image(dim_pix=244, im=TC_lab, location=location, im_or_mask=f"TC_lab", number=f"Region_{region}",
            #             folder=images_path)
            # split_image(dim_pix=244, im=GT, location=location, im_or_mask=f"Mask_lab", number=f"Region_{region}",
            #             folder=masks_path)


        except:
            continue
