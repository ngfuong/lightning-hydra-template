import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class DatasetDeepFashion2(Dataset):
    def __init__(self, datapath, transforms, split):
        self.datapath = datapath
        self.img_path = os.path.join(self.datapath, self.split, "image")
        self.transforms = transforms
        self.split = "val" if split in ["val", "test"] else "train"
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata["image_name"])

    def __getitem__(self, idx):
        # TODO: check this if data per batch is not loaded correctly
        triplet_paths, triplet_boxes, triplet_ids = self.sample(idx)
        triplet_imgs = self.load_frames(triplet_paths, boxes=triplet_boxes)
        # print("after load frames")
        # len(triplet_imgs)= 3 ; type(triplet_imgs) = list
        # type(triplet_imgs[0])=PIL Image
        triplet_imgs = [self.transforms(imgs) for imgs in triplet_imgs]
        # print(triplet_imgs[0].shape=(3, 224, 224))

        # batch = {
        # 'images': triplet_imgs, # list(torch.tensor(3, 224, 224)*3)
        # 'ids': triplet_ids, # nd.array([np.int64, np.int64, np.int64])
        # }
        return triplet_imgs, triplet_ids

    def load_frames(self, paths, boxes=None):
        # load images from paths
        # if boxes, crop corresponding box (x1,y1,x2,y2) from image
        frames = []
        for path, box in zip(paths, boxes):
            frames.append(self.read_image(path, box=box))
        return frames

    def read_image(self, image_name, box=None):
        r"""Return RGB image in PIL Image"""
        image = Image.open(os.path.join(self.img_path, image_name))
        if box is not None:
            image = image.crop(box)
        return image.convert("RGB")

    def sample(self, idx):
        """Sample path of triplet of images from the dataset"""
        # a_path, p_path, n_path = self.img_metadata['image_name'].iloc[idx].values
        paths = self.img_metadata["image_name"].iloc[idx].values
        # a_box, p_box, n_box = [self.img_metadata['box'].iloc[idx].values for _ in range(3)]
        boxes = self.img_metadata["box"].iloc[idx].values.reshape((3, -1))
        ids = self.img_metadata["category_id"].iloc[idx].values
        return paths, boxes, ids

    def build_class_ids(self):
        """Build a dictionary of class ids"""
        class_ids = {}
        for i, class_name in enumerate(self.df["class_name"].unique()):
            class_ids[class_name] = i
        return class_ids

    def build_img_metadata(self):
        def read_metadata(df):
            """Return metadata
            metadata: Dictionary
                image_name
                box
                category_id
            """
            metadata = {
                "image_name": df[["image_name_a", "image_name_p", "image_name_n"]],
                "box": df[
                    [
                        "x_1_a",
                        "y_1_a",
                        "x_2_a",
                        "y_2_a",
                        "x_1_p",
                        "y_1_p",
                        "x_2_p",
                        "y_2_p",
                        "x_1_n",
                        "y_1_n",
                        "x_2_n",
                        "y_2_n",
                    ]
                ],
                "category_id": df[["category_id_a", "category_id_p", "category_id_n"]],
            }
            return metadata

        def load_csv(split):
            with open(os.path.join(self.datapath, split + "_triplets.csv"), "r") as f:
                df = pd.read_csv(f)
                return df

        df = load_csv(self.split)

        img_metadata = {}
        if self.split in ["train", "val"]:
            img_metadata.update(read_metadata(df))
        else:
            raise Exception("Undefined split %s: " % self.split)

        print(
            "Total (%s) images are: %d" % (self.split, len(img_metadata["image_name"]))
        )
        return img_metadata


class DatasetDeepFashion(Dataset):
    """ Custom dataset with triplet sampling, for the Deep Fashion"""

    def _pil_loader(element):
        path = element[0]
        x1,y1,x2,y2 = element[1],element[2],element[3],element[4]
        img = Image.open(path)
        crop = img.crop((x1,y1,x2,y2))
        
        return crop.convert('RGB')

    def __init__(self, datapath, transforms, split, loader = _pil_loader):
        """
        Args:
            df: Dataframe
            root_dir (string): Directory with all the images.
            im_size (tuple): image size 
            train (boolean): True if create train set, False if test set
            transform (callable, optional): Optional transform to be applied
                on a sample.
            loader: function to load image
        Return:
            Dataset
        """
        self.split = "val" if split in ["val", "test"] else "train"
        self.datapath = datapath
        self.img_path = os.path.join(self.datapath, self.split, "image")
        self.transform = transforms
        self.loader = loader
        self.df = self.load_csv(self.split)
            
        self.df['image_name_a'] = self.df['image_pair_name_a'].apply(lambda x: os.path.join(self.datapath, x))
        self.df['image_name_p'] = self.df['image_pair_name_p'].apply(lambda x: os.path.join(self.datapath, x))
        self.df['image_name_n'] = self.df['image_pair_name_n'].apply(lambda x: os.path.join(self.datapath, x))
        
    def _sample(self,idx):
        p1 = self.df.loc[idx, ['image_name_a','x_1_a','y_1_a','x_2_a','y_2_a']].values.tolist()
        p2 = self.df.loc[idx, ['image_name_p','x_1_p','y_1_p','x_2_p','y_2_p']].values.tolist()
        p3 = self.df.loc[idx, ['image_name_n','x_1_n','y_1_n','x_2_n','y_2_n']].values.tolist()

        return [p1, p2, p3]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        paths = self._sample(idx)
        images = []
        for i in paths:
            temp = self.loader(i)
            temp = self.transform(temp)
            images.append(temp)
        return (images[0], images[1], images[2]),0

    def load_csv(self, split):
        with open(os.path.join(self.datapath, split + "_triplets.csv"), "r") as f:
            df = pd.read_csv(f)
            return df

