import cv2
import numpy as np
# import metrics
import matplotlib.pyplot as plt
# from metrics import *
import seaborn as sns
from sklearn.metrics import f1_score, precision_score
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
import sklearn
import numpy as np
# from metrics import mean_accuracy, pixel_accuracy, mean_IU


def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)

    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_


def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)

        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_


def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_


def frequency_weighted_IU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)

    sum_k_t_k = get_pixel_area(eval_segm)

    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_


'''
Auxiliary functions used during evaluation.
'''

def mean(list_of_items):
    return sum(list_of_items) / len(list_of_items)

def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")

def crop(array_to_crop, array_to_match):
    dims_to_match = array_to_match.shape
    return array_to_crop[0:dims_to_match[0], 0:dims_to_match[1]]


def mean(list_of_items):
    return sum(list_of_items) / len(list_of_items)


def evaluate(data, ground_truth, thresh, index_name):
    try:
        print(f"Evaluating {location}, {index_name}")
        data[data > thresh] = 1
        data[data < thresh] = 0

        print(np.amax(data))
        print(np.amax(ground_truth))

        print("MIoU: ", round(mean_IU(eval_segm=data, gt_segm=ground_truth), 4))
        # print("pa: ", round(pixel_accuracy(eval_segm=data, gt_segm=ground_truth), 4))
        # print("ma: ", round(mean_accuracy(eval_segm=data, gt_segm=ground_truth), 4))

        # print("AUC: ", round(sklearn.metrics.auc(data, ground_truth), 4))

        ground_truth[ground_truth == 1] = 2

        comb = ground_truth + data

        TN = np.count_nonzero(comb == 0)
        FN = np.count_nonzero(comb == 1)
        FP = np.count_nonzero(comb == 2)
        TP = np.count_nonzero(comb == 3)

        print("Rate of false positives: ", round(FP / (FP + TN), 4))
        print("Rate of false Negatives: ", round(FN / (FN + TP), 4))

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = (2 * precision * recall)/(precision + recall)
        print("F1 Score", round(F1, 4))

        cmap = colors.ListedColormap(['darkslategray',  # True Negative
                                      'orange',
                                      'lightcoral',  # False Negative
                                       # False Positive
                                      'cadetblue'])  # True positive


        bounds = [0, 1, 2, 3]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        plt.imsave(f"/Users/tj/PycharmProjects/eval/semantic-segmentation/test_data/{location}/{location}_{index_name}.png",
                   comb,
                   cmap=cmap)

        ground_truth[ground_truth == 2] = 1
        print("Evalutation complete")
        print(" ")
        print("________________________")
        print(" ")
    except:
        pass



if __name__ == "__main__":

    location = "Shanghai"
    snow = False

    # Extract Ground truth
    GT_path = f"/Users/tj/PycharmProjects/eval/semantic-segmentation/test_data/{location}_GT.png"
    GT = cv2.imread(GT_path)
    GT = np.array(GT)

    # Select one channel
    GT = GT[:, :, -1]

    # Generate a binary mask
    GT[GT > 0] = 1

    # Import/Extract MSI bands
    #
    MSI_path = f"/Users/tj/PycharmProjects/Semantic-Segmentation_DeepLabV3/Water_Mask_Eval/Plots/{location}/{location}_test_site.npy"
    #MSI_path = f"/Users/tj/PycharmProjects/Semantic-Segmentation_DeepLabV3/Water_Mask_Eval/Plots/Shanghai/Shanghai_cloud_test.npy"
    #MSI_path = f"/Users/tj/PycharmProjects/Semantic-Segmentation_DeepLabV3/Water_Mask_Eval/Plots/New_York/New_York_Snow_Test.npy"
    MSI = np.load(MSI_path)
    blue = crop(MSI[-1][:, :, 1], GT)
    green = crop(MSI[-1][:, :, 2], GT)
    red = crop(MSI[-1][:, :, 3], GT)
    NIR = crop(MSI[-1][:, :, 7], GT)
    SWIR1 = crop(MSI[-1][:, :, 10], GT)
    SWIR2 = crop(MSI[-1][:, :, 11], GT)



    NWDI_name = "NWDI"
    NWDI = (green - NIR)/(green + NIR)
    sns.distplot(NWDI, hist=False)
    plt.title(NWDI_name + f": {location}")
    plt.show()

    NWDI_snow_thres = 0.27

    if location == "Florida":
        NWDI_thresh = 0.1
    elif location == "New_York":
        NWDI_thresh = 0.1
    elif location == "Shanghai":
        NWDI_thresh = 0.2
    else:
        print("No location")

    if snow:
        evaluate(data=NWDI, ground_truth=GT, thresh=NWDI_snow_thres, index_name=NWDI_name)
    else:
        evaluate(data=NWDI, ground_truth=GT, thresh=NWDI_thresh, index_name=NWDI_name)




    MNDWI_name = "MNDWI"
    MNDWI = (green - SWIR2)/(green + SWIR2)
    sns.distplot(MNDWI, hist=False)
    plt.title(MNDWI_name + f": {location}")
    plt.show()

    MNDWI_snow_thres = 0.27

    if location == "Florida":
        MNDWI_thresh = 0.2
    elif location == "New_York":
        MNDWI_thresh = 0.25
    elif location == "Shanghai":
        MNDWI_thresh = 0.25
    else:
        print("No location")
    if snow:
        evaluate(data=MNDWI, ground_truth=GT, thresh=MNDWI_thresh, index_name=MNDWI_name)
    else:
        evaluate(data=MNDWI, ground_truth=GT, thresh=MNDWI_thresh, index_name=MNDWI_name)



    I_name = "I"
    I = ((green - NIR)/(green + NIR)) + ((blue - NIR)/(blue + NIR))
    sns.distplot(I, hist=False)
    plt.title(I_name + f": {location}")
    plt.show()

    I_snow_thres = 0.72
    if location == "Florida":
        I_thresh = 0.2
    elif location == "New_York":
        I_thresh = 0.15
    elif location == "Shanghai":
        I_thresh = 0.4
    else:
        print("No location")

    if snow:
        evaluate(data=I, ground_truth=GT, thresh=I_thresh, index_name=I_name)
    else:
        evaluate(data=I, ground_truth=GT, thresh=I_thresh, index_name=I_name)


    PI_name = "PI"
    PI = ((green - SWIR2)/(green + SWIR2)) + ((blue - NIR)/(blue + NIR))
    sns.distplot(PI, hist=False)
    plt.title(PI_name + f": {location}")
    plt.show()

    PI_snow_thres = 1
    if location == "Florida":
        PI_thresh = 0.25
    elif location == "New_York":
        PI_thresh = 0.5
    elif location == "Shanghai":
        PI_thresh = 0.45
    else:
        print("No location")

    if snow:
        evaluate(data=PI, ground_truth=GT, thresh=PI_snow_thres, index_name=PI_name)
    else:
        evaluate(data=PI, ground_truth=GT, thresh=PI_thresh, index_name=PI_name)


    AWEInsh_name = "AWEInsh"

    AWEInsh = (4 * (green - SWIR1) - (0.25 * NIR + 2.75 * SWIR2))/(green + NIR + SWIR1 + SWIR2)
    sns.distplot(AWEInsh, hist=False)
    plt.title(AWEInsh_name + f": {location}")
    plt.show()

    AWEInsh_snow_thres = 1.9
    if location == "Florida":
        AWEInsh_thresh = 0.8
    elif location == "New_York":
        AWEInsh_thresh = 0
    elif location == "Shanghai":
        AWEInsh_thresh = 1
    else:
        print("No location")
    if snow:
        evaluate(data=AWEInsh, ground_truth=GT, thresh=AWEInsh_snow_thres, index_name=AWEInsh_name)
    else:
        evaluate(data=AWEInsh, ground_truth=GT, thresh=AWEInsh_thresh, index_name=AWEInsh_name)

    AWEIsh_name = "AWEIsh"
    AWEIsh = (blue + 2.5 * green -1.5 * (NIR + SWIR1) - 0.25 * SWIR2)/(blue + green + NIR + SWIR1 + SWIR2)
    sns.distplot(AWEIsh, hist=False)
    plt.title(AWEIsh_name + f": {location}")
    plt.show()
    AWEIsh_snow_thres = 0.7
    if location == "Florida":
        AWEIsh_thresh = 0.5
    elif location == "New_York":
        AWEIsh_thresh = 0
    elif location == "Shanghai":
        AWEIsh_thresh = 0.7
    else:
        print("No location")
    if snow:
        evaluate(data=AWEIsh, ground_truth=GT, thresh=AWEIsh_snow_thres, index_name=AWEIsh_name)
    else:
        evaluate(data=AWEIsh, ground_truth=GT, thresh=AWEIsh_thresh, index_name=AWEIsh_name)




# PROPOSED WATER INDEX

    if location == "Florida":
        scalar = 0.03
    elif location == "New_York":
        scalar = 1.3
    elif location == "Shanghai":
        scalar = 1.37
    else:
        print("No location")
    PWI_snow_thres = 0.7

    #  0.9334

    # Proposed Water Index
    PWI_name = "PWI"


    #Florida
    PWI = -0.4 * ((SWIR2 - NIR) / (SWIR2 + NIR)) + ((green - NIR)/(green + NIR)) + ((blue - NIR)/(blue + NIR)) - (-0.2 * ((NIR - red) / (NIR + red)))


    # Shanghai
    # PWI = 0.5 * ((SWIR2 - NIR) / (SWIR2 + NIR)) + (4 * (green - SWIR1) - (0.25 * NIR + 2.75 * SWIR2))/(green + NIR + SWIR1 + SWIR2) - (3.4 * ((NIR - red) / (NIR + red)))

    # New York
    #PWI = 0.1 * ((SWIR2 - NIR) / (SWIR2 + NIR)) +(green - NIR)/(green + NIR)  - (0.3 * ((NIR - red) / (NIR + red)))

    a = (np.amax(PWI))
    b = (np.amin(PWI))
    print(a - b / 100)

    font = {'family': 'avenir',
            'color': 'Black',
            'weight': 'normal',
            'size': 17,
            }

    sns.distplot(PWI, hist=False, color="black", bins=100)
    plt.xlabel("Pixel Intensity value (Unitless)", fontdict=font)
    plt.ylabel("Probability per 1.4 interval in \n intensity value", fontdict=font)
    plt.title(f"{location}", fontdict=font)
    plt.savefig(f"/Users/tj/PycharmProjects/eval/semantic-segmentation/test_data/{location}/{location}_{PWI_name}_Intensity_Plot.png", bbox_inches="tight")
    plt.show()
    if location == "Florida":
        PWI_thresh = 0.2
    elif location == "New_York":
        PWI_thresh = 0.23
    elif location == "Shanghai":
        PWI_thresh = 0.9
    else:
        print("No location")
    evaluate(data=PWI, ground_truth=GT, thresh=PWI_thresh, index_name=PWI_name)

