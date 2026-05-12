import os
import shutil

# Source: The model we just verified
src_dir_global_user = r"C:\Users\dev\.EasyOCR\user_network"
src_dir_global_model = r"C:\Users\dev\.EasyOCR\model"

# Destination: Project folder
dst_project_net = r"C:\Users\dev\cleaning\backend\user_network"

def copy_v2():
    if not os.path.exists(dst_project_net):
        os.makedirs(dst_project_net)
        print(f"Created {dst_project_net}")

    # Files to copy
    # .pth from model folder
    shutil.copy2(os.path.join(src_dir_global_model, "ko_webtoon_v2.pth"), 
                 os.path.join(dst_project_net, "ko_webtoon_v2.pth"))
    
    # .py and .yaml from user_network folder
    shutil.copy2(os.path.join(src_dir_global_user, "ko_webtoon_v2.py"), 
                 os.path.join(dst_project_net, "ko_webtoon_v2.py"))
    
    shutil.copy2(os.path.join(src_dir_global_user, "ko_webtoon_v2.yaml"), 
                 os.path.join(dst_project_net, "ko_webtoon_v2.yaml"))
    
    print("Copied ko_webtoon_v2 files to project user_network folder.")

if __name__ == "__main__":
    copy_v2()
