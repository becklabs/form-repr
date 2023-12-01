from mmpose.apis import MMPoseInferencer
from typing import List, Dict

import argparse
import os


def extract_video_keypoints(
    video_path: str, inferencer: MMPoseInferencer, is_3d: bool = False
) -> List[Dict]:
    
    result_generator = inferencer(video_path, show=False)
    raw_results = [r for r in result_generator]
    results = []
    for frame in raw_results:
        if is_3d:
            if frame["predictions_2d"]:
                result_2d = frame["predictions_2d"][0]
            else:
                result_2d = []
            if frame["predictions"]:
                result_3d = frame["predictions"][0]
            else:
                result_3d = []
            if frame['predictions_2d']:
                result_2d = frame['predictions_2d'][0]
            else:
                result_2d = []
            if frame['predictions']:
                result_3d = frame['predictions'][0]
            else:
                result_3d = []
        else:
            if frame["predictions"]:
                result_2d = frame["predictions"][0]
            else:
                result_2d = []
            result_3d = []

        for subject in result_2d:
            subject["bbox_score"] = float(subject["bbox_score"])

        results.append({"predictions_3d": result_3d, "predictions_2d": result_2d})
    return results


def main(input_dir: str, output_dir: str, is_3d: bool, overwrite: bool) -> None:
    inferencer: MMPoseInferencer
    if is_3d:
        from inferencers import inferencer_3d

        inferencer = inferencer_3d
    else:
        from inferencers import inferencer_2d

        inferencer = inferencer_2d

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video_name in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_name)
        output_path = os.path.join(output_dir, video_name.replace(".mp4", ".json"))
        if not overwrite and os.path.exists(output_path):
            continue
        results = extract_video_keypoints(video_path, inferencer, is_3d)
        import json

        with open(output_path, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keypoints from a video")
    parser.add_argument("--video_path", type=str, help="path to video file")
    parser.add_argument("--output_path", type=str, help="path to output file")
    parser.add_argument(
        "--is_3d", type=bool, default=False, help="whether to extract 3d keypoints"
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help="whether to overwrite existing output file",
    )
    args = parser.parse_args()
    print(args)

    main(args.video_path, args.output_path, args.is_3d, args.overwrite)
