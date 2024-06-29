from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from pathlib import Path
from tqdm.auto import tqdm
import deep_sort.deep_sort.deep_sort as ds
import gradio as gr
import os

# 设置环境变量以避免 OpenMP 多次初始化错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 控制处理流程是否终止
should_continue = True


def get_detectable_classes(model_file):
    """获取给定模型文件可以检测的类别。"""
    model = YOLO(model_file)
    class_names = list(model.names.values())
    del model
    return class_names


def stop_processing():
    global should_continue
    should_continue = False
    return "尝试终止处理..."


def start_processing(input_path, output_path, detect_class, model, progress=gr.Progress(track_tqdm=True)):
    global should_continue
    should_continue = True

    detect_class = int(detect_class)
    model = YOLO(model)
    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")

    # 创建一个临时目录
    temp_dir = tempfile.mkdtemp()
    temp_output_path = os.path.join(temp_dir, 'output.avi')
    output_video_path = detect_and_track(input_path, temp_output_path, detect_class, model, tracker)

    # 转码为 MP4
    final_output_path = os.path.join(output_path, "output.mp4")
    ffmpeg_path = "D:\\tools\\ffmpeg-7.0.1-essentials_build\\bin\\ffmpeg.exe"  # 替换为FFmpeg的实际路径
    ffmpeg_command = f"{ffmpeg_path} -y -i {temp_output_path} -vcodec libx264 {final_output_path}"
    os.system(ffmpeg_command)

    # 删除临时文件
    os.remove(temp_output_path)

    return final_output_path, final_output_path


def putTextWithBackground(
        img,
        text,
        origin,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=1,
        text_color=(255, 255, 255),
        bg_color=(0, 0, 0),
        thickness=1,
):
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    bottom_left = origin
    top_right = (origin[0] + text_width, origin[1] - text_height - 5)
    cv2.rectangle(img, bottom_left, top_right, bg_color, -1)
    text_origin = (origin[0], origin[1] - 5)
    cv2.putText(
        img,
        text,
        text_origin,
        font,
        font_scale,
        text_color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def extract_detections(results, detect_class):
    detections = np.empty((0, 4))
    confarray = []
    for r in results:
        for box in r.boxes:
            if box.cls[0].int() == detect_class:
                x1, y1, x2, y2 = box.xywh[0].int().tolist()
                conf = round(box.conf[0].item(), 2)
                detections = np.vstack((detections, np.array([x1, y1, x2, y2])))
                confarray.append(conf)
    return detections, confarray


def detect_and_track(input_path: str, output_path: str, detect_class: int, model, tracker) -> Path:
    global should_continue
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output_video_path = Path(output_path)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_video = cv2.VideoWriter(output_video_path.as_posix(), fourcc, fps, size, isColor=True)

    for _ in tqdm(range(total_frames)):
        if not should_continue:
            print('stopping process')
            break

        success, frame = cap.read()
        if not success:
            break

        results = model(frame, stream=True)
        detections, confarray = extract_detections(results, detect_class)
        resultsTracker = tracker.update(detections, confarray, frame)

        for x1, y1, x2, y2, Id in resultsTracker:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            putTextWithBackground(frame, str(int(Id)), (max(-10, x1), max(40, y1)), font_scale=1.5,
                                  text_color=(255, 255, 255), bg_color=(255, 0, 255))

        output_video.write(frame)

    output_video.release()
    cap.release()

    print(f'output dir is: {output_video_path}')
    return output_video_path


if __name__ == "__main__":
    model_list = ["best.pt"]
    detect_classes = get_detectable_classes(model_list[0])
    examples = [["test.mp4", tempfile.mkdtemp(), detect_classes[0], model_list[0]], ]

    with gr.Blocks() as demo:
        with gr.Tab("Tracking"):
            gr.Markdown(
                """
                # 目标检测与跟踪
                基于opencv + YoloV8 + deepsort
                """
            )
            with gr.Row():
                with gr.Column():
                    input_path = gr.Video(label="Input video")
                    model = gr.Dropdown(model_list, value=model_list[0], label="Model")
                    detect_class = gr.Dropdown(detect_classes, value=detect_classes[0], label="Class", type='index',
                                               allow_custom_value=True)
                    output_dir = gr.Textbox(label="Output dir", value=tempfile.mkdtemp())
                    with gr.Row():
                        start_button = gr.Button("Process")
                        stop_button = gr.Button("Stop")
                with gr.Column():
                    output = gr.Video()
                    output_path = gr.Textbox(label="Output path")

                    gr.Examples(examples, label="Examples",
                                inputs=[input_path, output_dir, detect_class, model],
                                outputs=[output, output_path],
                                fn=start_processing,
                                cache_examples=False)

        start_button.click(start_processing, inputs=[input_path, output_dir, detect_class, model],
                           outputs=[output, output_path])
        stop_button.click(stop_processing)

    demo.launch()
