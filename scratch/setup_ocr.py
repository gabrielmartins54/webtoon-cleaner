import os
import shutil

# Paths
src_model = r"C:\Users\dev\easyOCR\EasyOCR\trainer\saved_models\ko_webtoon_v2\best_accuracy.pth"
src_py = r"C:\Users\dev\easyOCR\EasyOCR\user_network\ko_webtoon.py"
dst_user_network = r"C:\Users\dev\.EasyOCR\user_network"
dst_model_dir = r"C:\Users\dev\.EasyOCR\model"

# Targets
dst_model = os.path.join(dst_model_dir, "ko_webtoon_v2.pth")
dst_yaml = os.path.join(dst_user_network, "ko_webtoon_v2.yaml")
dst_py = os.path.join(dst_user_network, "ko_webtoon_v2.py")

# Characters from opt.txt
character = '0123456789!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~  !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEHIJKLMNOPRSTWY[\]^_`adefghijkmnorstuvwxy{|}~※ⓒㄴㅃㅋ가각감갑개거걱건걸검것게겠격경계고공관괄괴교구국군권귀그글금급기까깎뀌끝나난날났낱내냐너네넷녕노놈높뇌누느늑는능니님닝다단담답당대더던데도돌동돼되된될됩두뒷드든들등디딥따때떻뜸라람래러런럴럽렇레려력로루류률르를릇리릴림마만말맞머먹멘면모목못무물뭐뭔뮤므미민및바박반받발방배버번범법벨별보복본부분브비빨뿐사살상색샘생서선설성세소속송수술숨스슨습승시신심싶쎄쓰씨아안알앗앤야약어얼업없었에역연열염였영오올와왁완왔외요용우움웅워원유윤으은을음의이익인일읽임입있자작잖잠장재저적전접정제조좀좋주죽준중즈증지직진집찝차창채처천첩체초총촥추출충츠치칭카커켜코콰쿠크키킬타탄탈태터테템토통투튜트틀티파패팰퍼페폄포폭폰표푸품프픈플피하한할합해행향허헐현혈형혜호확환회획후흐흡히힘'

yaml_content = f"""network_params:
  input_channel: 1
  output_channel: 256
  hidden_size: 256
imgH: 64
lang_list:
- ko
character_list: {character}
"""

def setup():
    for d in [dst_user_network, dst_model_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created {d}")

    # Copy model to 'model' folder
    shutil.copy2(src_model, dst_model)
    print(f"Copied model to {dst_model}")

    # Copy py to 'user_network'
    shutil.copy2(src_py, dst_py)
    print(f"Copied py definition to {dst_py}")

    # Write yaml to 'user_network'
    with open(dst_yaml, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"Created yaml at {dst_yaml}")

if __name__ == "__main__":
    setup()
