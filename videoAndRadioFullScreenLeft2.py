# OpenCV版本的视频检测（全屏优化版）- 修复中文显示和语法错误
import cv2
from keras.models import load_model
import numpy as np
import datetime
import time
import threading
import queue
import os
from PIL import Image, ImageDraw, ImageFont  # 导入PIL库用于中文渲染
import speech

# 初始化语音消息队列和锁
speech_queue = queue.Queue(maxsize=3)
speech_lock = threading.Lock()

# 加载情绪分类模型
emotion_classifier = load_model('classifier/emotion_models/simple_CNN.530-0.65.hdf5')

emotion_labels = {
    0: '生气',
    1: '厌恶',
    2: '恐惧',
    3: '开心',
    4: '难过',
    5: '惊喜',
    6: '平静'
}

# 中文到英文的文件名映射
emotion_file_mapping = {
    '生气': 'angry',
    '厌恶': 'disgust',
    '恐惧': 'fear',
    '开心': 'happy',
    '难过': 'sad',
    '惊喜': 'surprise',
    '平静': 'calm'
}

# 正确加载OpenCV提供的人脸检测级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 创建一个线程安全的变量来存储最新的情绪状态
class LatestEmotion:
    def __init__(self):
        self.lock = threading.Lock()
        self.emotion = None
        self.timestamp = None

    def set(self, emotion, timestamp):
        with self.lock:
            self.emotion = emotion
            self.timestamp = timestamp

    def get(self):
        with self.lock:
            return self.emotion, self.timestamp

latest_emotion = LatestEmotion()

# 使用绝对路径加载表情图片和反馈图片
def load_emotion_images():
    """加载cunxin文件夹中的表情图片"""
    emotion_images = {}
    emotion_folder = r"img\cunxin"
    
    # 确保文件夹存在
    if not os.path.exists(emotion_folder):
        print(f"警告: 表情图片文件夹不存在: {emotion_folder}")
        print("当前工作目录:", os.getcwd())
        return emotion_images
    
    for emotion in emotion_labels.values():
        # 使用英文文件名
        english_name = emotion_file_mapping.get(emotion, emotion)
        img_path = os.path.join(emotion_folder, f"{english_name}.png")
        
        if os.path.exists(img_path):
            # 使用OpenCV读取图片，处理中文路径问题
            image_data = np.fromfile(img_path, dtype=np.uint8)
            emotion_img = cv2.imdecode(image_data, cv2.IMREAD_UNCHANGED)
            
            if emotion_img is not None:
                emotion_img = cv2.resize(emotion_img, (90, 90))
                emotion_images[emotion] = emotion_img
                print(f"成功加载表情图片: {img_path}")
            else:
                print(f"无法加载图片，可能是文件损坏: {img_path}")
                emotion_images[emotion] = None
        else:
            print(f"图片不存在: {img_path}")
            emotion_images[emotion] = None
    
    return emotion_images

def load_feedback_images():
    """加载img文件夹中的反馈图片"""
    feedback_images = {}
    img_folder = r"img"
    
    # 确保文件夹存在
    if not os.path.exists(img_folder):
        print(f"警告: 反馈图片文件夹不存在: {img_folder}")
        print("当前工作目录:", os.getcwd())
        return feedback_images
    
    # 加载正确图片 (使用英文文件名)
    correct_path = os.path.join(img_folder, "right.png")
    if os.path.exists(correct_path):
        # 使用OpenCV读取图片，处理中文路径问题
        image_data = np.fromfile(correct_path, dtype=np.uint8)
        correct_img = cv2.imdecode(image_data, cv2.IMREAD_UNCHANGED)
        
        if correct_img is not None:
            correct_img = cv2.resize(correct_img, (90, 90))
            feedback_images['correct'] = correct_img
            print(f"成功加载正确图片: {correct_path}")
        else:
            print(f"无法加载正确图片，可能是文件损坏: {correct_path}")
            feedback_images['correct'] = None
    else:
        print(f"正确图片不存在: {correct_path}")
        feedback_images['correct'] = None
    
    # 加载错误图片 (使用英文文件名)
    wrong_path = os.path.join(img_folder, "error.png")
    if os.path.exists(wrong_path):
        # 使用OpenCV读取图片，处理中文路径问题
        image_data = np.fromfile(wrong_path, dtype=np.uint8)
        wrong_img = cv2.imdecode(image_data, cv2.IMREAD_UNCHANGED)
        
        if wrong_img is not None:
            wrong_img = cv2.resize(wrong_img, (100, 100))
            feedback_images['wrong'] = wrong_img
            print(f"成功加载错误图片: {wrong_path}")
        else:
            print(f"无法加载错误图片，可能是文件损坏: {wrong_path}")
            feedback_images['wrong'] = None
    else:
        print(f"错误图片不存在: {wrong_path}")
        feedback_images['wrong'] = None
    
    return feedback_images

# 加载所有图片
emotion_images = load_emotion_images()
feedback_images = load_feedback_images()

# 设置初始目标表情
target_emotion = '开心'

# 中文字体设置 - 使用PIL绘制中文
def setup_chinese_font():
    """设置中文字体"""
    # 尝试多种常见中文字体路径
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux文泉驿微米黑
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                # 测试字体是否可用
                font = ImageFont.truetype(font_path, 30)
                print(f"使用中文字体: {font_path}")
                return font_path
            except:
                print(f"字体加载失败: {font_path}")
                continue
    
    print("警告: 未找到可用的中文字体，中文显示可能不正常")
    return None

# 获取中文字体路径
chinese_font_path = setup_chinese_font()

def put_chinese_text(img, text, position, font_size, color):
    """
    使用PIL在图像上绘制中文文本
    """
    if chinese_font_path is None:
        # 如果没有中文字体，使用OpenCV的默认渲染（会显示问号）
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size/30, color, 2)
        return img
    
    # 将OpenCV图像转换为PIL格式
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 加载中文字体
    font = ImageFont.truetype(chinese_font_path, font_size)
    
    # 绘制中文文本
    draw.text(position, text, font=font, fill=color)
    
    # 转换回OpenCV格式
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

def speech_worker():
    """语音播放线程的工作函数"""
    last_spoken_emotion = None
    while True:
        current_emotion, timestamp = latest_emotion.get()
        if current_emotion is None:
            time.sleep(0.01)
            continue

        if current_emotion != last_spoken_emotion and current_emotion is not None:
            with speech_lock:
                try:
                    speech.say(current_emotion)
                    last_spoken_emotion = current_emotion
                    print(f"语音播放: {current_emotion} (时间戳: {timestamp})")
                except Exception as e:
                    print(f"语音播放出错: {e}")
        else:
            time.sleep(0.05)

# 创建并启动语音线程
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def draw_left_panel(img, target_emotion, detected_emotion):
    """
    在图像左侧绘制表情信息面板 - 使用PIL绘制中文
    """
    height, width = img.shape[:2]
    
    # 左侧面板宽度（占整个宽度的25%，更宽松的布局）
    panel_width = int(width * 0.25)
    
    # 创建左侧面板的背景（深灰色）
    panel_bg = np.zeros((height, panel_width, 3), dtype=np.uint8)
    panel_bg[:] = (60, 60, 60)  # 深灰色背景
    
    # 面板上半部分：显示表情文字
    text_area_height = int(height * 0.4)
    
    # 使用PIL绘制中文文本
    panel_bg = put_chinese_text(panel_bg, "目标表情", (10, 10), 25, (255, 255, 255))
    panel_bg = put_chinese_text(panel_bg, target_emotion, (10, 50), 30, (0, 255, 255))
    
    # 显示检测到的表情
    panel_bg = put_chinese_text(panel_bg, "检测表情", (10, text_area_height - 40), 25, (255, 255, 255))
    detected_text = detected_emotion if detected_emotion else "未检测到"
    text_color = (0, 255, 0) if detected_emotion == target_emotion else (0, 0, 255)
    panel_bg = put_chinese_text(panel_bg, detected_text, (10, text_area_height), 30, text_color)
    
    # 面板下半部分：显示表情图片
    if target_emotion in emotion_images and emotion_images[target_emotion] is not None:
        emotion_img = emotion_images[target_emotion]
        img_h, img_w = emotion_img.shape[:2]
        
        # 计算图片位置（居中显示）
        img_x = (panel_width - img_w) // 2
        img_y = text_area_height + (height - text_area_height - img_h) // 2
        
        # 确保坐标在范围内
        img_x = max(0, min(img_x, panel_width - img_w))
        img_y = max(text_area_height, min(img_y, height - img_h))
        
        # 处理透明图片
        if len(emotion_img.shape) == 3 and emotion_img.shape[2] == 4:
            alpha_channel = emotion_img[:, :, 3] / 255.0
            for c in range(0, 3):
                panel_bg[img_y:img_y+img_h, img_x:img_x+img_w, c] = (
                    alpha_channel * emotion_img[:, :, c] +
                    (1 - alpha_channel) * panel_bg[img_y:img_y+img_h, img_x:img_x+img_w, c]
                )
        else:
            # 处理不透明图片
            panel_bg[img_y:img_y+img_h, img_x:img_x+img_w] = emotion_img
    else:
        # 如果图片加载失败，显示错误信息
        panel_bg = put_chinese_text(panel_bg, "图片加载失败", (10, text_area_height + 100), 20, (0, 0, 255))
    
    # 将面板合并到原始图像左侧
    img[0:height, 0:panel_width] = panel_bg
    
    return panel_width

def draw_feedback_icon(img, is_correct):
    """在右上角绘制反馈图标"""
    height, width = img.shape[:2]
    
    feedback_type = 'correct' if is_correct else 'wrong'
    feedback_img = feedback_images.get(feedback_type)
    
    if feedback_img is not None:
        img_h, img_w = feedback_img.shape[:2]
        
        # 右上角位置
        img_x = width - img_w - 20
        img_y = 20
        
        # 确保坐标在范围内
        img_x = max(0, min(img_x, width - img_w))
        img_y = max(0, min(img_y, height - img_h))
        
        # 处理透明图片
        if len(feedback_img.shape) == 3 and feedback_img.shape[2] == 4:
            alpha_channel = feedback_img[:, :, 3] / 255.0
            for c in range(0, 3):
                img[img_y:img_y+img_h, img_x:img_x+img_w, c] = (
                    alpha_channel * feedback_img[:, :, c] +
                    (1 - alpha_channel) * img[img_y:img_y+img_h, img_x:img_x+img_w, c]
                )
        else:
            # 处理不透明图片
            img[img_y:img_y+img_h, img_x:img_x+img_w] = feedback_img

def discern(img, frame_timestamp, target_emotion):
    """
    在图像中检测人脸并识别情绪
    """
    current_emotion_for_frame = None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 人脸检测
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # 提取人脸区域
            gray_face = gray[y:y+h, x:x+w]
            
            try:
                # 预处理人脸图像用于情绪识别
                gray_face = cv2.resize(gray_face, (48, 48))
                gray_face = gray_face / 255.0
                gray_face = np.expand_dims(gray_face, axis=0)
                gray_face = np.expand_dims(gray_face, axis=-1)

                # 情绪预测
                emotion_prediction = emotion_classifier.predict(gray_face, verbose=0)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion = emotion_labels[emotion_label_arg]
                current_emotion_for_frame = emotion

            except Exception as e:
                print(f"情绪预测出错: {e}")
                emotion = "未知"

            # 绘制人脸框和情绪标签
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            try:
                # 使用PIL绘制中文情绪标签
                img = put_chinese_text(img, emotion, (x, y-30), 25, (0, 255, 0))
            except Exception as e:
                print(f"添加文本出错: {e}")
                # 备用方案：使用OpenCV绘制英文
                cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 更新最新情绪状态
    latest_emotion.set(current_emotion_for_frame, frame_timestamp)

    # 绘制左侧面板
    panel_width = draw_left_panel(img, target_emotion, current_emotion_for_frame)
    
    # 检查表情是否匹配并绘制反馈图标
    if current_emotion_for_frame is not None:
        is_match = current_emotion_for_frame == target_emotion
        draw_feedback_icon(img, is_match)
    
    # 显示完整图像（包含左侧面板和摄像头画面）
    cv2.imshow("Face and Emotion Detection", img)
    
    return img

# 主程序
def main():
    global target_emotion
    
    # 获取摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return
    
    # 创建窗口并设置为全屏模式
    cv2.namedWindow("Face and Emotion Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Face and Emotion Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # 强制窗口位置到屏幕左上角，避免窗口显示在屏幕外
    cv2.moveWindow("Face and Emotion Detection", 0, 0)
    
    frame_counter = 0
    process_every_n_frame = 1
    prev_time = time.time()
    fps = 0
    
    try:
        while True:
            ret, img = cap.read()
            if not ret:
                print("错误: 无法读取摄像头帧")
                break
            
            frame_counter += 1
            current_time = time.time()
            frame_timestamp = current_time
            
            # 计算并显示FPS
            if current_time - prev_time > 0:
                fps = 1.0 / (current_time - prev_time)
            prev_time = current_time
            
            # 在图像上添加FPS信息（放在右侧避免被左侧面板覆盖）
            cv2.putText(img, f"FPS: {fps:.1f}", (img.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if frame_counter % process_every_n_frame == 0:
                discern(img, frame_timestamp, target_emotion)
            else:
                latest_emotion.set(None, frame_timestamp)
                panel_width = draw_left_panel(img, target_emotion, None)
                cv2.imshow("Face and Emotion Detection", img)
            
            # 键盘输入处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                current_state = cv2.getWindowProperty("Face and Emotion Detection", cv2.WND_PROP_FULLSCREEN)
                if current_state == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty("Face and Emotion Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty("Face and Emotion Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            elif key == ord('t'):
                # 切换目标表情
                emotions_list = list(emotion_labels.values())
                if target_emotion in emotions_list:
                    current_index = emotions_list.index(target_emotion)
                    target_emotion = emotions_list[(current_index + 1) % len(emotions_list)]
                    print(f"目标表情已切换为: {target_emotion}")
                else:
                    # 如果当前目标表情不在列表中，默认选择第一个
                    target_emotion = emotions_list[0]
                    print(f"目标表情已重置为: {target_emotion}")
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()