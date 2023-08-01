import argparse
import cv2
import reader
import image_enchant
import utils

def helper(img) -> str:
    try: 
        passport = reader.Back(img)
        return passport.MRZ
    except UnboundLocalError:
        return 'bad image'

def main(image_path):
    img = cv2.imread(image_path)
    images = image_enchant.get_card(img)
    mrz = []
    for image in images:
        m = helper(image)
        if utils.validate_mrz(m):
            mrz.append(m)
        m = helper(image_enchant.rotate_image(image, 180))
        if utils.validate_mrz(m):
            mrz.append(m)

    if len(mrz) == 0:
        print("не удалось корректно обработать изображение")
    else:
        print(mrz[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Path to the image")
    args = parser.parse_args()
    main(args.image)
