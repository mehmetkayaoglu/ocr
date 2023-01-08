import argparse
import os
import string

import torch
import torchvision

import dataset
from model import CRNN

# if cuda is available use cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def main(args):
    # Required model input image format (width x height x channel) is (100 x 32 x 1)
    imgW = 100
    imgH = 32
    number_of_channels = 1

    # Create ocr alphabet
    alphabet = ''.join(string.ascii_uppercase + "0123456789")
    number_of_class = 1 + len(alphabet)
    print(f"number_of_class: {number_of_class}")
    converter = StrLabelConverter(alphabet)

    # Create output directory, if not exist
    os.makedirs(args.output_path, exist_ok=True)

    """
    Prepare data
    """
    # Prepare transforms for resizing and normalization of input image
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((imgH, imgW)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Training data
    train_dataset = dataset.CustomDataset(root=args.train_image_folder, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate)

    # Validation data
    val_dataset = dataset.CustomDataset(root=args.validation_image_folder, transform=transform)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate)

    """
    Prepare model
    """
    model = CRNN(imgH, number_of_channels, number_of_class)
    model.to(device)
    model.apply(weights_init)

    criterion = torch.nn.CTCLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)

    """
    Run for each epoch
    """
    for epoch in range(args.num_epochs):
        print(f'epoch {epoch}:')

        """
        Train
        """
        # Put model to training stage
        model.train()

        # Iterate over all training data
        for i, (images, texts) in enumerate(train_dataloader):
            # Get the batch size. Last batch maybe smaller than batch_size.
            batch_size = images.size(0)

            # Encode the text
            text, length = converter.encode(texts)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            predictions = model(images.to(device))
            predictions_size = torch.autograd.Variable(torch.LongTensor([predictions.size(0)] * batch_size))
            cost = criterion(predictions, text, predictions_size, length) / batch_size
            cost.backward()
            optimizer.step()

            # print statistics
            if i % 10 == 0:
                print(f"[{epoch}/{args.num_epochs}][{i}/{len(train_dataloader)}] Loss: {cost:.8f}")

        """
        Validation
        """
        n_correct = 0
        val_losses = []

        # Put model to evaluation stage
        model.eval()

        # Iterate over all validation data
        for i, (images, texts) in enumerate(val_dataloader):
            # Get the batch size. Last batch maybe smaller than batch_size.
            batch_size = images.size(0)

            # Encode the text
            text, length = converter.encode(texts)

            predictions = model(images.to(device))
            predictions_size = torch.autograd.Variable(torch.LongTensor([predictions.size(0)] * batch_size))
            cost = criterion(predictions, text, predictions_size, length) / batch_size
            val_losses.append(cost)

            _, predictions = predictions.max(2)
            predictions = predictions.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(predictions.data, predictions_size.data, raw=False)

            cpu_texts_decode = []
            for text in texts:
                cpu_texts_decode.append(text.decode('utf-8', 'strict'))

            for pred, target in zip(sim_preds, cpu_texts_decode):
                if pred == target:
                    n_correct += 1

        # Print statistics
        raw_preds = converter.decode(predictions.data, predictions_size.data, raw=True)[:10]
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts_decode):
            print(f"{raw_pred:20} => {pred:20}, gt: {gt:20}")
        accuracy = n_correct / float(i * batch_size)
        print(f"Validation loss: {torch.utils.data}, accuray: {accuracy}")

        # Save model at every 10 epochs
        if epoch % 10 == 0:
            save_file_name = f'ocr_model-{epoch}.pth'
            torch.save(model.state_dict(), os.path.join(args.output_path, save_file_name))


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
    parser.add_argument('-t', '--train_image_folder', type=str, required=True, help='train image folder')
    parser.add_argument('-v', '--validation_image_folder', type=str, required=True, help='validation image folder')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='output path')
    parser.add_argument('-n', '--num_epochs', type=int, default=250, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')
    main(parser.parse_args())
