import argparse
import glob
import os

import readcard

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog="ReadCard",
        description="Read the data of an IBM 80 column punched card from an image"
    )
    arg_parser.add_argument("deckname", help="The directory containing the deck of files to analyze.")
    arg_parser.add_argument("-t", "--threshold", type=int, default=127, help="The threshold value used to distinguish the card from the background.")
    arg_parser.add_argument("-k", "--kernelsize", type=int, default=5, help="The size of the dilate/erode kernel.")
    arg_parser.add_argument("-m", "--mediansize", type=int, default=5, help="The size of the median filter kernel.")
    args = arg_parser.parse_args()

    files = glob.glob(os.path.join(args.deckname, "*"))
    for file in files:
      print(readcard.read_card(file, args.threshold, args.kernelsize, args.mediansize))
