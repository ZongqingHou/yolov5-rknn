import glob
import shutil

root_path = "/home/hdd/volums/yolov5-rknn/extract20241101/colorring/images"
dest_path = "/home/hdd/volums/yolov5-rknn/extract20241101/colorring/images-merge"

def filter(search_path, buffer):
    search_result = glob.glob("{}/*.jpg".format(search_path))
    search_result.sort()

    if search_result:
        for tmp_data in search_result:
            tmp_path = tmp_data.replace(root_path, "")
            tmp_file_name = "_".join(tmp_path.split("/"))[1:]
            shutil.copy(tmp_data, "{}/{}".format(dest_path, tmp_file_name))
            print(tmp_file_name)

    else:
        tmp_collections = glob.glob("{}/*".format(search_path))
        tmp_collections.sort()

        for tmp_data in tmp_collections:
            filter(tmp_data, buffer)


if __name__ == "__main__":
    import argparse

    parse = argparse.ArgumentParser(description="create realsence depth list")
    parse.add_argument("--dataset", default="/home/hdd/volums/yolov5-rknn/extract20241101/colorring/images", type=str)
    parse.add_argument("--dest_path", default="/home/hdd/volums/yolov5-rknn/extract20241101/colorring/images.txt",
                       type=str)

    opt = parse.parse_args()

    with open(opt.dest_path, "w") as buffer:
        filter(opt.dataset, buffer)
