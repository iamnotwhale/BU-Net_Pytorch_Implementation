import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm
import argparse

def n4itk_bias_correction(image):
    """
    Applies N4ITK bias correction to the given image.
    """
    image = sitk.Cast(image, sitk.sitkFloat32)
    mask_image = sitk.OtsuThreshold(image, 0, 1, 200)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(image, mask_image)
    return corrected_image

def normalize_image(image):
    """
    Normalizes the image to zero mean and unit variance.
    """
    image_array = sitk.GetArrayFromImage(image)

    # Discard top 1% and bottom 1%
    lower_percentile = np.percentile(image_array, 1)
    upper_percentile = np.percentile(image_array, 99)
    image_array = np.clip(image_array, lower_percentile, upper_percentile)

    # Normalize to zero mean and unit variance
    mean = np.mean(image_array)
    std = np.std(image_array)
    normalized_array = (image_array - mean) / std

    normalized_image = sitk.GetImageFromArray(normalized_array)
    normalized_image.CopyInformation(image)
    return normalized_image

def preprocess_image(image_path, output_path):
    """
    Preprocesses the image: applies N4ITK bias correction and normalization.
    """
    image = sitk.ReadImage(image_path)

    corrected_image = n4itk_bias_correction(image)
    normalized_image = normalize_image(corrected_image)

    sitk.WriteImage(normalized_image, output_path)


def main(folder, save_path):
    for image in tqdm(sorted(os.listdir(folder))):
        # print(image, os.listdir(folder+image))
        if not os.path.exists(save_path + image):
            os.makedirs(save_path + image)
        if image == '.DS_Store':
            continue
        for f in os.listdir(folder+image):
            if 'seg' in f:
                continue
            else:
                input_image_path = os.path.join(folder, image, f)
                output_image_path = os.path.join(save_path, image, f)
                preprocess_image(input_image_path, output_image_path)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Preprocess images")

    argparser.add_argument("--folder", type=str, required=True, help="Path to the folder containing images")
    argparser.add_argument("--save_path", type=str, required=True, help="Path to save the preprocessed images")
    
    args = argparser.parse_args()
    main(args.folder, args.save_path)