import cv2
import pandas as pd
import os
from pathlib import Path
import json
import shutil


def split_to_train_test(df, label_column, train_frac=0.8):
    # stratified split
    train_df, test_df = pd.DataFrame(), pd.DataFrame()
    labels = df[label_column].unique()
    for lbl in labels:
        lbl_df = df[df[label_column] == lbl]
        lbl_train_df = lbl_df.sample(frac=train_frac)
        lbl_test_df = lbl_df.drop(lbl_train_df.index)
        train_df = pd.concat([train_df, lbl_train_df])
        test_df = pd.concat([test_df, lbl_test_df])
    return train_df, test_df


def processing(pc_base_dir):
    # find the images sizes
    directory = pc_base_dir / "dataset" / "images.txt"
    images_df = pd.read_csv(
        directory, sep=" ", names=["id", "image_file_name"], header=None
    ).dropna(axis=0)
    images_sizes = []
    directory = pc_base_dir / "dataset" / "images"
    for id, ifn in zip(images_df["id"], images_df["image_file_name"]):
        img = cv2.imread(directory / ifn)
        if img is None:
            raise ValueError("Error in the image path.")
        height, width, _ = img.shape
        image_size = {"id": id, "width": width, "height": height}
        images_sizes.append(image_size)
    sizes_df = pd.DataFrame(images_sizes)

    directory = pc_base_dir / "dataset" / "bounding_boxes.txt"
    bboxes_df = pd.read_csv(
        directory,
        sep=" ",
        names=["id", "x_abs", "y_abs", "bbox_width", "bbox_height"],
        header=None,
    )
    directory = pc_base_dir / "dataset" / "image_class_labels.txt"
    image_class_labels_df = pd.read_csv(
        directory, sep=" ", names=["id", "class_id"], header=None
    )

    # merge all the metadata into one dataframe
    # images_df = images_df.reset_index()
    full_df = pd.merge(images_df, image_class_labels_df, on="id")
    full_df = pd.merge(full_df, sizes_df, on="id")
    full_df = pd.merge(full_df, bboxes_df, on="id")
    # full_df.sort_values(by=["index"], inplace=True)

    # Define the bounding boxes in the format required by SageMaker's built in Object Detection algorithm.
    # the xmin/ymin/xmax/ymax parameters are specified as ratios to the total image pixel size
    full_df["xmin"] = full_df["x_abs"] / full_df["width"]
    full_df["xmax"] = (full_df["x_abs"] + full_df["bbox_width"]) / full_df["width"]
    full_df["ymin"] = full_df["y_abs"] / full_df["height"]
    full_df["ymax"] = (full_df["y_abs"] + full_df["bbox_height"]) / full_df["height"]

    # drop the columns that are not needed
    full_df.drop(
        columns=["x_abs", "y_abs", "bbox_width", "bbox_height"],
        inplace=True,
    )

    # small subset of species to reduce resources consumption
    smaple_classes = list(map(int, os.environ["SAMPLE_CLASSES"].split(",")))
    criteria = full_df["class_id"].isin(smaple_classes)
    full_df = full_df[criteria]

    # split the data into train and test
    train_df, test_df = split_to_train_test(full_df, "class_id", train_frac=0.8)

    # save files
    train_dir = pc_base_dir / "train"
    train_dir.mkdir(exist_ok=True)
    valid_dir = pc_base_dir / "validation"
    valid_dir.mkdir(exist_ok=True)

    # now we have right data. we just put them in the json file as required.
    def create_json_dict(df):
        json_dict = {"images": [], "annotations": []}
        for _, row in df.iterrows():
            # image info
            image_info = {
                "file_name": row["image_file_name"],
                "height": int(row["height"]),
                "width": int(row["width"]),
                "id": int(row["id"]),
            }
            json_dict["images"].append(image_info)

            # annotation info
            bbox = [
                float(row["xmin"]),
                float(row["ymin"]),
                float(row["xmax"]),
                float(row["ymax"]),
            ]
            annotation = {
                "image_id": int(row["id"]),
                "bbox": bbox,
                "category_id": int(row["class_id"]),
            }
            json_dict["annotations"].append(annotation)
        return json_dict

    with open(train_dir / "train.json", "w") as f:
        json.dump(create_json_dict(train_df), f)
    with open(valid_dir / "validation.json", "w") as f:
        json.dump(create_json_dict(test_df), f)

    # model input feed directory
    model_input_dir = pc_base_dir / "model_input_feed"
    model_input_dir.mkdir(exist_ok=True)
    images_dir = model_input_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # put the images
    source_dir = pc_base_dir / "dataset" / "images"
    for image_file in train_df["image_file_name"]:
        src_path = source_dir / image_file
        dst_path = images_dir / image_file
        print(f"Checking: {src_path} → {dst_path}")  # Debugging line
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
    # put the json file
    shutil.copy2(train_dir / "train.json", model_input_dir / "train.json")


if __name__ == "__main__":
    pc_base_dir = Path(os.environ["PC_BASE_DIR"])
    processing(pc_base_dir)
