"""fiftyone ソフトの動作確認
MS COCO のデータセットを整理して確認できる
"""
import fiftyone as fo
import fiftyone.zoo as foz

def fiftyone_main():
    dataset = foz.load_zoo_dataset("quickstart")
    session = fo.launch_app(dataset)
    session.wait()

if __name__ == "__main__":
    fiftyone_main()