import argparse
import os
import string

import torch
import torchvision
from PIL import Image

from model import CRNN

# if cuda is available use cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    # Required model input image format (width x height x channel) is (100 x 32 x 1)
    imgW = 100
    imgH = 32
    number_of_channels = 1

    # Create ocr alphabet
    alphabet = ''.join(string.ascii_uppercase + "0123456789")
    number_of_class = 1 + len(alphabet)
    converter = StrLabelConverter(alphabet)

    # Load model
    model = CRNN(imgH, number_of_channels, number_of_class)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # Put model to evaluation stage
    model.eval()

    # Prepare transforms for resizing and normalization of input image
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((imgH, imgW)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Get image file names from the input image directory
    image_names = os.listdir(args.image_folder)

    # Create output directory, if not exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Predict and write results to the file
    with open(args.output_path, "w") as file:

        # Run for each image in the directory
        for image_name in image_names:
            print(f"Processing image '{image_name}'")

            # Read image as gray level
            image_path = os.path.join(args.image_folder, image_name)
            image = Image.open(image_path).convert('L')

            # Apply transforms to the input image
            image = transform(image)
            image = image.view(1, *image.size())

            # Make prediction
            prediction = model(image.to(device))
            _, prediction = prediction.max(2)
            prediction = prediction.transpose(1, 0).contiguous().view(-1)
            prediction_size = torch.autograd.Variable(torch.LongTensor([prediction.size(0)]))
            prediction = converter.decode(prediction.data, prediction_size.data, raw=False)

            # Write to the file
            # file.write(f"{image_name} {sim_pred}\n")
            file.write(f"{os.path.splitext(image_name)[0]} {prediction}\n")


class StrLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.LongTensor [n]: length of each text.
        """

        length = []
        result = []
        for item in text:
            item = item.decode('utf-8', 'strict')
            length.append(len(item))
            r = []
            for char in item:
                index = self.dict[char]
                r.append(index)
            result.append(r)

        max_len = 0
        for r in result:
            if len(r) > max_len:
                max_len = len(r)

        result_temp = []
        for r in result:
            for i in range(max_len - len(r)):
                r.append(0)
            result_temp.append(r)

        text = result_temp
        return (torch.LongTensor(text), torch.LongTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.LongTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.LongTensor([l]), raw=raw))
                index += l
            return texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_folder', type=str, required=True, help='image folder')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='model path')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='output path')
    main(parser.parse_args())
