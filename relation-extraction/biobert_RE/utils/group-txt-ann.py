import os
import shutil
from os.path import abspath, join, exists

if __name__ == "__main__":

    in_dir = abspath("../datasets/acs-20210114-ungrouped")
    out_dir = abspath("../datasets/acs-20210114")

    pub_nums = set()
    for item in os.listdir(in_dir):
        pub_nums.add(item.split(".")[0])

    for pub_num in pub_nums:
        article_out_dir = join(out_dir, pub_num)
        if not exists(article_out_dir):
            os.mkdir(article_out_dir)
        shutil.copy(join(in_dir, f"{pub_num}.ann"), article_out_dir)
        shutil.copy(join(in_dir, f"{pub_num}.txt"), article_out_dir)
    