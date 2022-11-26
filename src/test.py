import argparse
import torch

from src.models.online_triplet_module import OnlineTripletModule
import torchvision.transforms as transform


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Visual Search Test Script")
        # model and train hyperparam
        parser.add_argument(
            "--model",
            type=str,
            default="vs-model",
            help="model name (default: vs-model)"
        )
        parser.add_argument(
            "--backbone",
            type=str,
            default="resnet50",
            help="backbone for feature extraction (default: resnet50)"
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="shopping100k",
            help="dataset name for evaluation (default: shopping100k)"
        )
        parser.add_argument(
            "--data-path",
            type=str,
            default="data/shopping100k/",
            help="path to test image folder"
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=16,
            metavar="N",
            help="number of test dataloader threads"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for testing (default: auto"
        )
        # cuda, seed
        parser.add_argument(
            "--no-cuda",
            action="store_true",
            default=False,
            help="disables CUDA training",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=1,
            metavar="S",
            help="random seed (default: 1)"
        )
        # test option
        parser.add_argument(
            "--weights",
            type=int,
            default=None,
            help="path to checkpoint weights to test (default: 1)"
        )

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(f"===========\n{args}\n=============")


def test(args):
    module = OnlineTripletModule.load_from_checkpoint(
        checkpoint_path=args.weights,
        data_path = args.data_path,
        dataset = args.dataset,
        backbone = args.backbone,
        # num_features=512,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
    )

    

    input_transform = transform.Compose([
        
    ])