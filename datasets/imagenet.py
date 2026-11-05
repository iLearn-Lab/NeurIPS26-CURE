import os
import pickle
from collections import OrderedDict

from datasets.utils import *
from .oxford_pets import OxfordPets


class ImageNet(DatasetBase):

    def __init__(self, root, shot, seed, subsample="all"):
        self.dataset_dir = os.path.join(root, "ImageNet")
        self.image_dir = os.path.join(self.dataset_dir, "data")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            test = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = shot
        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, train_u= data["train"], data["unlabel_train"]
            else:
                train, train_u = self.generate_fewshot_dataset(train, num_shots=num_shots, unlabel=True)
                data = {"train": train, 'unlabel_train': train_u}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train,train_u=train_u, val=test, test=test)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        from .imagenet_classes import imagenet_classes
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.split(",")[0].split()
                folder = line[0]
                classname = " ".join(line[1:])
                # classnames[folder] = classname
                classnames[folder] = imagenet_classes[i]
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)

        # skip the duplicated classes
        skip = ['n04008634', 'n04355933']
        
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir() and f.name not in skip)
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
