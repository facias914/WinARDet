import base64
import io
import json
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Union, Set, Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask


class BaseRewardFunction(ABC):
    """Base reward function abstract class"""

    @abstractmethod
    def compute_reward(self, predict: str, ground_truth: str) -> float:
        """Compute reward score"""
        pass

    @abstractmethod
    def get_reward_name(self) -> str:
        """Get reward function name"""
        pass


class BoxIoURewardFunction(BaseRewardFunction):
    """Bounding box IoU reward function"""

    def get_reward_name(self) -> str:
        return "box_iou"

    def parse_ground_truth(self, ground_truth: str) -> Optional[Dict]:
        """Parse ground truth string, extract image dimensions and annotation information"""
        answer = ground_truth["answer"]

        # 使用resized_image_size进行坐标转换
        resized_size = ground_truth["resized_image_size"]
        width, height = resized_size

        # 提取对象信息
        objects = {}

        for class_name, class_data in answer.items():
            if "boxes" in class_data:
                objects[class_name] = class_data["boxes"]

        return {"dims": (width, height), "objects": objects, "raw_data": answer}

    def parse_detection_output(
        self, text: str, width: int, height: int
    ) -> Optional[Dict]:
        """解析模型输出字符串，提取检测到的对象"""
        try:
            text = text.replace("\n", "").strip()

            objects = {}
            pattern = (
                r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
                r"<\|box_start\|>(.*?)<\|box_end\|>"
            )

            matches = re.findall(pattern, text)

            for class_name, boxes_str in matches:
                class_name = class_name.strip()
                if class_name not in objects:
                    objects[class_name] = []

                # 查找所有边界框坐标
                box_pattern = r"<(\d+)>"
                all_coords = re.findall(box_pattern, boxes_str)

                # 将坐标分组为4个一组 (x1, y1, x2, y2)
                for i in range(0, len(all_coords), 4):
                    if i + 3 < len(all_coords):
                        try:
                            x1, y1, x2, y2 = [
                                int(coord) for coord in all_coords[i : i + 4]
                            ]

                            # 将归一化坐标转换为实际像素坐标
                            x1_abs = x1 / 1000.0 * width
                            y1_abs = y1 / 1000.0 * height
                            x2_abs = x2 / 1000.0 * width
                            y2_abs = y2 / 1000.0 * height

                            # 确保有效的边界框 (x1 < x2, y1 < y2)
                            x1_final = min(x1_abs, x2_abs)
                            y1_final = min(y1_abs, y2_abs)
                            x2_final = max(x1_abs, x2_abs)
                            y2_final = max(y1_abs, y2_abs)

                            # 跳过退化的边界框（零面积）
                            if x1_final < x2_final and y1_final < y2_final:
                                objects[class_name].append(
                                    [x1_final, y1_final, x2_final, y2_final]
                                )
                        except (ValueError, IndexError):
                            continue

            return {"objects": objects}
        except Exception:
            return None

    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个边界框的交并比"""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def compute_reward(self, predict: str, ground_truth: str) -> float:
        """计算边界框IoU奖励分数"""
        # 解析ground truth获取图像尺寸
        gt_data = self.parse_ground_truth(ground_truth)
        if gt_data is None:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- Reward: {ground_truth['reward_name']}: 1.0 | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                    )
                    f.write(f"Fail to parse ground truth: {ground_truth}\n")
                    f.write(f"Prediction: {predict}\n\n")
            return 0.0

        width, height = gt_data["dims"]

        # 使用ground truth的尺寸解析预测
        pred_data = self.parse_detection_output(predict, width, height)
        if pred_data is None:
            self._log_debug(f"Failed to parse prediction: {predict}")
            return 0.0

        gt_objects = gt_data["objects"]
        pred_objects = pred_data["objects"]

        # 收集所有边界框
        all_gt_boxes = [
            (box, class_name)
            for class_name, boxes in gt_objects.items()
            for box in boxes
        ]
        all_pred_boxes = [
            (box, class_name)
            for class_name, boxes in pred_objects.items()
            for box in boxes
        ]

        num_gt = len(all_gt_boxes)
        num_pred = len(all_pred_boxes)

        if num_gt == 0 and num_pred == 0:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- Reward: {ground_truth['reward_name']}: 1.0 | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                    )
                    f.write(f"Full Rejection and pred is None. GT: {ground_truth}\n")
                    f.write(f"Prediction: {predict}\n\n")
            return 1.0

        if num_gt == 0 and num_pred != 0:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- Reward: {ground_truth['reward_name']}: 1.0 | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                    )
                    f.write(
                        f"Full Rejection and pred is not None. GT: {ground_truth}\n"
                    )
                    f.write(f"Prediction: {predict}\n\n")
            return 0.0
        # 计算Recall
        total_recall_score = 0.0
        for gt_box, gt_class in all_gt_boxes:
            best_iou = 0.0
            for pred_box, pred_class in all_pred_boxes:
                if gt_class == pred_class:
                    iou = self.calculate_iou(gt_box, pred_box)
                    best_iou = max(best_iou, iou)
            total_recall_score += best_iou

        recall = total_recall_score / num_gt if num_gt > 0 else 0.0

        # 计算Precision
        total_precision_score = 0.0
        for pred_box, pred_class in all_pred_boxes:
            best_iou = 0.0
            for gt_box, gt_class in all_gt_boxes:
                if pred_class == gt_class:
                    iou = self.calculate_iou(pred_box, gt_box)
                    best_iou = max(best_iou, iou)
            total_precision_score += best_iou

        precision = total_precision_score / num_pred if num_pred > 0 else 0.0

        # 计算F1分数
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"------------- Reward: {ground_truth['reward_name']}: {f1_score} | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                )
                f.write(f"Precision: {precision}, Recall: {recall}\n\n")
                f.write(f"Prediction: {predict}\n")
                f.write(f"GT: {json.dumps(ground_truth['answer'])}\n")
                # f.write(f"Solution: {ground_truth['answer']}\n\n")

        # 可视化功能
        visualize_path = os.getenv("LOG_VISUALIZE_PATH")
        if visualize_path:
            try:
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                dataset_name = ground_truth.get("dataset_name", "unknown")
                save_path = (
                    f"{visualize_path}/box_iou_{dataset_name}_{current_time}.png"
                )

                if ensure_dir_exists(save_path):
                    create_visualization(
                        gt_data=gt_data,
                        pred_data=pred_data,
                        reward_score=f1_score,
                        reward_name="box_iou",
                        dataset_name=dataset_name,
                        save_path=save_path,
                    )
            except Exception as e:
                print(f"Visualization failed for box_iou: {e}")

        return f1_score


class PointInBoxRewardFunction(BaseRewardFunction):
    """点是否在框内的奖励函数"""

    def get_reward_name(self) -> str:
        return "point_in_box"

    def parse_ground_truth(self, ground_truth: str) -> Optional[Dict]:
        """解析ground truth字符串，提取图像尺寸和点框对应信息"""
        answer = ground_truth["answer"]

        # 使用resized_image_size进行坐标转换
        resized_size = ground_truth["resized_image_size"]
        width, height = resized_size

        # 提取对象信息 - 每个类别同时有points和boxes
        objects = {}

        for class_name, class_data in answer.items():
            if "points" in class_data and "boxes" in class_data:
                points = class_data["points"]
                boxes = class_data["boxes"]
                # 确保points和boxes长度一致
                if len(points) == len(boxes):
                    objects[class_name] = {"points": points, "boxes": boxes}

        return {"dims": (width, height), "objects": objects, "raw_data": answer}

    def parse_detection_output(
        self, text: str, width: int, height: int
    ) -> Optional[Dict]:
        """解析模型输出字符串，提取检测到的点和类别"""
        try:
            text = text.replace("\n", "").strip()

            objects = {}
            pattern = (
                r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
                r"<\|box_start\|>(.*?)<\|box_end\|>"
            )

            matches = re.findall(pattern, text)

            for class_name, points_str in matches:
                class_name = class_name.strip()
                if class_name not in objects:
                    objects[class_name] = []

                # 查找所有点坐标
                point_pattern = r"<(\d+)>"
                all_coords = re.findall(point_pattern, points_str)

                # 将坐标分组为2个一组 (x, y)
                for i in range(0, len(all_coords), 2):
                    if i + 1 < len(all_coords):
                        try:
                            x, y = [int(coord) for coord in all_coords[i : i + 2]]

                            # 将归一化坐标转换为实际像素坐标
                            x_abs = x / 1000.0 * width
                            y_abs = y / 1000.0 * height

                            objects[class_name].append([x_abs, y_abs])
                        except (ValueError, IndexError):
                            continue

            return {"objects": objects}
        except Exception:
            return None

    def is_point_in_box(self, point: List[float], box: List[float]) -> bool:
        """判断点是否在边界框内"""
        x, y = point
        x0, y0, x1, y1 = box

        # 确保边界框坐标顺序正确
        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)

        return x_min <= x <= x_max and y_min <= y <= y_max

    def compute_reward(self, predict: str, ground_truth: str) -> float:
        """计算点是否在框内的奖励分数"""
        # 解析ground truth获取图像尺寸和点框对应信息
        gt_data = self.parse_ground_truth(ground_truth)
        if gt_data is None:
            self._log_debug(f"Failed to parse ground truth: {ground_truth}")
            return 0.0

        width, height = gt_data["dims"]

        # 使用ground truth的尺寸解析预测
        pred_data = self.parse_detection_output(predict, width, height)
        if pred_data is None:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- Reward: {ground_truth['reward_name']}: 1.0 | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                    )
                    f.write(f"Fail to parse ground truth: {ground_truth}\n")
                    f.write(f"Prediction: {predict}\n\n")
            return 0.0

        gt_objects = gt_data["objects"]
        pred_objects = pred_data["objects"]

        # 收集所有ground truth的点框对和类别信息
        all_gt_point_box_pairs = []
        for class_name, class_data in gt_objects.items():
            points = class_data["points"]
            boxes = class_data["boxes"]
            for point, box in zip(points, boxes):
                all_gt_point_box_pairs.append((point, box, class_name))

        # 收集所有预测的点
        all_pred_points = []
        for class_name, points in pred_objects.items():
            for point in points:
                all_pred_points.append((point, class_name))

        num_gt = len(all_gt_point_box_pairs)
        num_pred = len(all_pred_points)

        if num_gt == 0 and num_pred == 0:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- Reward: {ground_truth['reward_name']}: 1.0 | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                    )
                    f.write(f"No gt and pred. GT: {ground_truth}\n")
                    f.write(f"Prediction: {predict}\n\n")
            return 1.0
        if num_gt == 0 or num_pred == 0:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- Reward: {ground_truth['reward_name']}: 0.0 | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                    )
                    f.write(f"No gt or pred. GT: {ground_truth}\n")
                    f.write(f"Prediction: {predict}\n\n")
            return 0.0

        # 贪心匹配：对于每个GT框，寻找一个点和他匹配
        # 一旦匹配上，该GT框和预测点就不要再参与匹配
        matched_gt_indices = set()
        matched_pred_indices = set()

        # 为每个GT框寻找最佳匹配的预测点
        for gt_idx, (gt_point, gt_box, gt_class) in enumerate(all_gt_point_box_pairs):
            if gt_idx in matched_gt_indices:
                continue

            best_match_score = 0.0
            best_match_idx = -1

            for pred_idx, (pred_point, pred_class) in enumerate(all_pred_points):
                if pred_idx in matched_pred_indices:
                    continue  # 这个预测点已经被匹配了

                # 判断预测点是否在GT框内
                if self.is_point_in_box(pred_point, gt_box):
                    # 如果类别一致，reward为1，否则为0
                    match_score = 1.0 if gt_class == pred_class else 0.0
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_idx = pred_idx

            # 如果找到匹配，标记为已匹配
            if best_match_idx >= 0:
                matched_gt_indices.add(gt_idx)
                matched_pred_indices.add(best_match_idx)

        # 计算Recall: 匹配成功的GT框数量 / 总GT数量
        recall = len(matched_gt_indices) / num_gt if num_gt > 0 else 0.0

        # 计算Precision: 匹配成功的预测点数量 / 总预测数量
        precision = len(matched_pred_indices) / num_pred if num_pred > 0 else 0.0

        # 计算F1分数
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"------------- Reward: {ground_truth['reward_name']}: {f1_score} | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                )
                f.write(f"Precision: {precision}, Recall: {recall}\n\n")
                f.write(
                    f"Matched GT: {len(matched_gt_indices)}, Total GT: {num_gt}, Matched Pred: {len(matched_pred_indices)}, Total Pred: {num_pred}\n\n"
                )
                f.write(f"Prediction: {predict}\n")
                f.write(f"GT: {json.dumps(ground_truth['answer'])}\n")
                # f.write(f"Solution: {ground_truth['answer']}\n\n")

        # 可视化功能
        visualize_path = os.getenv("LOG_VISUALIZE_PATH")
        if visualize_path:
            try:
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                dataset_name = ground_truth.get("dataset_name", "unknown")
                save_path = (
                    f"{visualize_path}/point_in_box_{dataset_name}_{current_time}.png"
                )

                if ensure_dir_exists(save_path):
                    create_visualization(
                        gt_data=gt_data,
                        pred_data=pred_data,
                        reward_score=f1_score,
                        reward_name="point_in_box",
                        dataset_name=dataset_name,
                        save_path=save_path,
                    )
            except Exception as e:
                print(f"Visualization failed for point_in_box: {e}")

        return f1_score


class PointInMaskRewardFunction(BaseRewardFunction):
    """点是否在mask内的奖励函数"""

    def get_reward_name(self) -> str:
        return "point_in_mask"

    def parse_ground_truth(self, ground_truth: str) -> Optional[Dict]:
        """解析ground truth字符串，提取图像尺寸和点mask对应信息"""
        answer = ground_truth["answer"]

        # 使用resized_image_size进行坐标转换
        resized_size = ground_truth["resized_image_size"]
        width, height = resized_size

        # 提取对象信息 - 每个类别同时有points和masks
        objects = {}

        for class_name, class_data in answer.items():
            if "points" in class_data and "masks" in class_data:
                points = class_data["points"]
                masks = class_data["masks"]
                # 确保points和masks长度一致
                if len(points) == len(masks):
                    objects[class_name] = {"points": points, "masks": masks}

        return {"dims": (width, height), "objects": objects, "raw_data": answer}

    def parse_detection_output(
        self, text: str, width: int, height: int
    ) -> Optional[Dict]:
        """解析模型输出字符串，提取检测到的点和类别"""
        try:
            text = text.replace("\n", "").strip()

            objects = {}
            pattern = (
                r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
                r"<\|box_start\|>(.*?)<\|box_end\|>"
            )

            matches = re.findall(pattern, text)

            for class_name, points_str in matches:
                class_name = class_name.strip()
                if class_name not in objects:
                    objects[class_name] = []

                # 查找所有点坐标
                point_pattern = r"<(\d+)>"
                all_coords = re.findall(point_pattern, points_str)

                # 将坐标分组为2个一组 (x, y)
                for i in range(0, len(all_coords), 2):
                    if i + 1 < len(all_coords):
                        try:
                            x, y = [int(coord) for coord in all_coords[i : i + 2]]

                            # 将归一化坐标转换为实际像素坐标
                            x_abs = x / 1000.0 * width
                            y_abs = y / 1000.0 * height

                            objects[class_name].append([x_abs, y_abs])
                        except (ValueError, IndexError):
                            continue

            return {"objects": objects}
        except Exception:
            return None

    def is_point_in_mask(
        self, point: List[float], mask: Union[dict, list], height: int, width: int
    ) -> bool:
        """判断点是否在mask内"""
        try:
            x, y = point

            # 确保坐标在有效范围内
            if x < 0 or x >= width or y < 0 or y >= height:
                return False

            # 解码mask
            if isinstance(mask, dict) and "counts" in mask:
                # RLE format

                binary_mask = coco_mask.decode(mask)
            elif isinstance(mask, list):
                # Already decoded mask
                binary_mask = np.array(mask)
            else:
                return False

            # 检查点是否在mask内
            if (
                binary_mask is not None
                and binary_mask.shape[0] == height
                and binary_mask.shape[1] == width
            ):
                return bool(binary_mask[int(y), int(x)])

            return False
        except Exception as e:
            if os.getenv("DEBUG_MODE") == "true":
                self._log_debug(f"Failed to check point in mask: {e}")
            return False

    def compute_reward(self, predict: str, ground_truth: str) -> float:
        """计算点是否在mask内的奖励分数"""
        # 解析ground truth获取图像尺寸和点mask对应信息
        gt_data = self.parse_ground_truth(ground_truth)
        if gt_data is None:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- Reward: {ground_truth['reward_name']}: 1.0 | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                    )
                    f.write(f"Fail to parse ground truth: {ground_truth}\n")
                    f.write(f"Prediction: {predict}\n\n")
            return 0.0

        width, height = gt_data["dims"]

        # 使用ground truth的尺寸解析预测
        pred_data = self.parse_detection_output(predict, width, height)
        if pred_data is None:
            self._log_debug(f"Failed to parse prediction: {predict}")
            return 0.0

        gt_objects = gt_data["objects"]
        pred_objects = pred_data["objects"]

        # 收集所有ground truth的点mask对和类别信息
        all_gt_point_mask_pairs = []
        for class_name, class_data in gt_objects.items():
            points = class_data["points"]
            masks = class_data["masks"]
            for point, mask in zip(points, masks):
                all_gt_point_mask_pairs.append((point, mask, class_name))

        # 收集所有预测的点
        all_pred_points = []
        for class_name, points in pred_objects.items():
            for point in points:
                all_pred_points.append((point, class_name))

        num_gt = len(all_gt_point_mask_pairs)
        num_pred = len(all_pred_points)

        if num_gt == 0 and num_pred == 0:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- Reward: {ground_truth['reward_name']}: 1.0 | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                    )
                    f.write(f"No gt and pred. GT: {ground_truth}\n")
                    f.write(f"Prediction: {predict}\n\n")
            return 1.0
        if num_gt == 0 or num_pred == 0:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- Reward: {ground_truth['reward_name']}: 0.0 | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                    )
                    f.write(f"No gt or pred. GT: {ground_truth}\n")
                    f.write(f"Prediction: {predict}\n\n")
            return 0.0

        # 贪心匹配：对于每个GT mask，寻找一个点和他匹配
        # 一旦匹配上，该GT mask和预测点就不要再参与匹配
        matched_gt_indices = set()
        matched_pred_indices = set()

        # 为每个GT mask寻找最佳匹配的预测点
        for gt_idx, (gt_point, gt_mask, gt_class) in enumerate(all_gt_point_mask_pairs):
            if gt_idx in matched_gt_indices:
                continue

            best_match_score = 0.0
            best_match_idx = -1

            for pred_idx, (pred_point, pred_class) in enumerate(all_pred_points):
                if pred_idx in matched_pred_indices:
                    continue  # 这个预测点已经被匹配了

                # 判断预测点是否在GT mask内
                if self.is_point_in_mask(pred_point, gt_mask, height, width):
                    # 如果类别一致，reward为1，否则为0
                    match_score = 1.0 if gt_class == pred_class else 0.0
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_idx = pred_idx

            # 如果找到匹配，标记为已匹配
            if best_match_idx >= 0:
                matched_gt_indices.add(gt_idx)
                matched_pred_indices.add(best_match_idx)

        # 计算Recall: 匹配成功的GT mask数量 / 总GT数量
        recall = len(matched_gt_indices) / num_gt if num_gt > 0 else 0.0

        # 计算Precision: 匹配成功的预测点数量 / 总预测数量
        precision = len(matched_pred_indices) / num_pred if num_pred > 0 else 0.0

        # 计算F1分数
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"------------- Reward: {ground_truth['reward_name']}: {f1_score} | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                )
                f.write(f"Precision: {precision}, Recall: {recall}\n\n")
                f.write(
                    f"Matched GT: {len(matched_gt_indices)}, Total GT: {num_gt}, Matched Pred: {len(matched_pred_indices)}, Total Pred: {num_pred}\n\n"
                )
                f.write(f"Prediction: {predict}\n")

        # 可视化功能
        visualize_path = os.getenv("LOG_VISUALIZE_PATH")
        if visualize_path:
            try:
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                dataset_name = ground_truth.get("dataset_name", "unknown")
                save_path = (
                    f"{visualize_path}/point_in_mask_{dataset_name}_{current_time}.png"
                )

                if ensure_dir_exists(save_path):
                    create_visualization(
                        gt_data=gt_data,
                        pred_data=pred_data,
                        reward_score=f1_score,
                        reward_name="point_in_mask",
                        dataset_name=dataset_name,
                        save_path=save_path,
                    )
            except Exception as e:
                print(f"Visualization failed for point_in_mask: {e}")

        return f1_score

    def _log_debug(self, message: str):
        """记录调试信息"""
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            if log_path:
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- {current_time} {self.get_reward_name()} reward -------------\n"
                    )
                    f.write(f"{message}\n\n")


class MosaicVQARewardFunction(BaseRewardFunction):
    """Quadrant-level presence reward for mosaic VQA images.

    Reward design:
      - Presence reward:
            For each quadrant q and category c in the mosaic candidate set:
                + alpha  if c is in GT(q) and predicted in q
                - beta   if c is NOT in GT(q) but predicted in q
                - gamma  if c is in GT(q) but NOT predicted in q

      - Wrong-repeat penalty:
            For categories that should NOT appear in quadrant q, repeated wrong
            predictions are additionally penalized:
                - delta * min(n_pred(q, c), tau)

        where n_pred(q, c) is the number of predicted boxes of category c that
        are assigned to quadrant q.

    Raw reward:
        R_raw = R_presence + R_wrong_repeat

    Normalization:
        First linearly map R_raw from [R_min, R_max] to [0, 1], then map to
        [-1, 1]:

            R_01   = (R_raw - R_min) / (R_max - R_min)
            R_norm = 2 * R_01 - 1

        with clipping to [-1, 1].

    Theoretical bounds:
        Let:
            Q = number of quadrants
            K = number of unique categories in the current mosaic
            S = sum_q |GT_categories(q)|

        Then:
            R_max = alpha * S

            R_min = - gamma * S - (beta + delta * tau) * (Q * K - S)

    Required ground_truth fields:
        - sub_images: list[dict], each dict contains:
            - top_left: [x1, y1]
            - bottom_right: [x2, y2]
            - categories: list[str]
        - resized_image_size: [width, height]
    """

    def __init__(
        self,
        lambda_hit: float = 1.0,
        lambda_fp: float = 1.5,
        lambda_fn: float = 1.0,
        lambda_leak: float = 0.5,
        repeat_cap: int = 3,
    ):
        """
        Keep the external constructor signature compatible as much as possible.

        Parameter mapping:
            lambda_hit  -> alpha   (reward for correct presence)
            lambda_fp   -> beta    (penalty for wrong category presence)
            lambda_fn   -> gamma   (penalty for missing GT category)
            lambda_leak -> delta   (extra penalty for repeated wrong predictions)
            repeat_cap  -> tau     (cap in wrong-repeat penalty)
        """
        self.lambda_hit = lambda_hit   # alpha
        self.lambda_fp = lambda_fp     # beta
        self.lambda_fn = lambda_fn     # gamma
        self.lambda_leak = lambda_leak # delta
        self.repeat_cap = repeat_cap   # tau

    def get_reward_name(self) -> str:
        return "mosaic_vqa"

    def _parse_pred_boxes(
        self, text: str, width: int, height: int
    ) -> Dict[str, List[List[float]]]:
        """Parse model output into:
            {class_name: [[x1, y1, x2, y2], ...]}
        in pixel coordinates.
        """
        text = text.replace("\n", "").strip()
        objects: Dict[str, List[List[float]]] = {}

        pattern = (
            r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
            r"<\|box_start\|>(.*?)<\|box_end\|>"
        )

        for class_name, boxes_str in re.findall(pattern, text):
            class_name = class_name.strip()
            if class_name not in objects:
                objects[class_name] = []

            all_coords = re.findall(r"<(\d+)>", boxes_str)
            for i in range(0, len(all_coords), 4):
                if i + 3 < len(all_coords):
                    try:
                        vals = [int(c) for c in all_coords[i : i + 4]]
                        x1 = vals[0] / 1000.0 * width
                        y1 = vals[1] / 1000.0 * height
                        x2 = vals[2] / 1000.0 * width
                        y2 = vals[3] / 1000.0 * height

                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)

                        if x1 < x2 and y1 < y2:
                            objects[class_name].append([x1, y1, x2, y2])
                    except (ValueError, IndexError):
                        continue

        return objects

    @staticmethod
    def _find_quadrant_by_center(
        box: List[float],
        quadrants: List[List[float]],
    ) -> int:
        """Assign a box to a quadrant by its center point.

        Returns:
            quadrant index if the center lies in a quadrant, else -1.
        """
        xc = 0.5 * (box[0] + box[2])
        yc = 0.5 * (box[1] + box[3])

        for qi, qr in enumerate(quadrants):
            x1, y1, x2, y2 = qr
            if x1 <= xc <= x2 and y1 <= yc <= y2:
                return qi
        return -1

    def _assign_to_quadrants(
        self,
        pred_objects: Dict[str, List[List[float]]],
        quadrants: List[List[float]],
    ) -> Dict[int, Dict[str, int]]:
        """Assign predicted boxes to quadrants by center point.

        Returns:
            {
                q_idx: {
                    class_name: count,
                    ...
                },
                ...
            }

        This preserves predicted box counts per (quadrant, class), which is
        needed for the wrong-repeat penalty.
        """
        quad_class_counts: Dict[int, Dict[str, int]] = {
            qi: {} for qi in range(len(quadrants))
        }

        for cls_name, boxes in pred_objects.items():
            for box in boxes:
                qi = self._find_quadrant_by_center(box, quadrants)
                if qi < 0:
                    continue
                quad_class_counts[qi][cls_name] = quad_class_counts[qi].get(cls_name, 0) + 1

        return quad_class_counts

    def compute_reward(self, predict: str, ground_truth: Dict[str, Any]) -> float:
        sub_images = ground_truth.get("sub_images", [])
        resized = ground_truth.get("resized_image_size", [1024, 1024])
        width, height = resized

        if not sub_images:
            return 0.0

        gt_cats_per_quad: List[Set[str]] = [
            set(s.get("categories", [])) for s in sub_images
        ]
        quadrant_regions: List[List[float]] = [
            s["top_left"] + s["bottom_right"] for s in sub_images
        ]

        all_gt_cats: Set[str] = set()
        for cats in gt_cats_per_quad:
            all_gt_cats.update(cats)

        # Parse predictions and assign boxes to quadrants
        pred_objects = self._parse_pred_boxes(predict, width, height)
        pred_counts_per_quad = self._assign_to_quadrants(pred_objects, quadrant_regions)

        # Presence reward + wrong-repeat penalty
        alpha = self.lambda_hit
        beta = self.lambda_fp
        gamma = self.lambda_fn
        delta = self.lambda_leak
        tau = self.repeat_cap

        presence_reward = 0.0
        wrong_repeat_penalty = 0.0

        # Optional debug bookkeeping
        quad_presence_breakdown: List[Dict[str, float]] = []

        for qi in range(len(sub_images)):
            gt_c = gt_cats_per_quad[qi]
            pred_count_dict = pred_counts_per_quad.get(qi, {})
            pred_c = set(pred_count_dict.keys())

            hit_count = 0
            wrong_count = 0
            miss_count = 0
            repeat_penalty_q = 0.0

            for c in all_gt_cats:
                gt_has = c in gt_c
                pred_has = pred_count_dict.get(c, 0) > 0

                if gt_has and pred_has:
                    presence_reward += alpha
                    hit_count += 1
                elif (not gt_has) and pred_has:
                    presence_reward -= beta
                    wrong_count += 1
                elif gt_has and (not pred_has):
                    presence_reward -= gamma
                    miss_count += 1

            # Extra penalty for repeated wrong-category predictions
            for c, n_pred in pred_count_dict.items():
                if c not in gt_c:
                    repeat_penalty_q += delta * min(n_pred, tau)

            wrong_repeat_penalty -= repeat_penalty_q

            quad_presence_breakdown.append(
                {
                    "hit": float(hit_count),
                    "wrong": float(wrong_count),
                    "miss": float(miss_count),
                    "repeat_penalty": float(repeat_penalty_q),
                }
            )

        raw_reward = presence_reward + wrong_repeat_penalty

        # Theoretical bounds for normalization
        Q = len(sub_images)
        K = len(all_gt_cats)
        S = sum(len(cats) for cats in gt_cats_per_quad)

        r_max = alpha * S
        r_min = -gamma * S - (beta + delta * tau) * (Q * K - S)

        if r_max > r_min:
            reward_01 = (raw_reward - r_min) / (r_max - r_min)
            normalized = 2.0 * reward_01 - 1.0
        else:
            normalized = 0.0

        reward = max(-1.0, min(1.0, normalized))

        # Debug logging
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- Reward: mosaic_vqa: {reward:.4f} | "
                        f"Dataset: {ground_truth.get('dataset_name', 'unknown')} "
                        f"-------------\n"
                    )
                    f.write(
                        f"Presence: {presence_reward:.2f}, "
                        f"WrongRepeat: {wrong_repeat_penalty:.2f}, "
                        f"Raw: {raw_reward:.2f}, "
                        f"R_min: {r_min:.2f}, "
                        f"R_max: {r_max:.2f}, "
                        f"Norm: {normalized:.4f}\n"
                    )
                    for qi in range(len(sub_images)):
                        pred_count_dict = pred_counts_per_quad.get(qi, {})
                        f.write(
                            f"  Q{qi}: GT={gt_cats_per_quad[qi]} "
                            f"PredCats={set(pred_count_dict.keys())} "
                            f"PredCounts={pred_count_dict} "
                            f"Stats={quad_presence_breakdown[qi]}\n"
                        )
                    f.write(f"Image: {ground_truth.get('image_name', '')}\n")
                    f.write(f"Prediction: {predict}\n\n")

        return reward


class RewardFunctionFactory:
    """奖励函数工厂类"""

    _reward_functions = {}

    @classmethod
    def register(cls, reward_name: str, reward_class: type):
        """注册奖励函数"""
        cls._reward_functions[reward_name] = reward_class

    @classmethod
    def get_reward_function(cls, reward_name: str) -> Optional[BaseRewardFunction]:
        """获取奖励函数实例"""
        if reward_name in cls._reward_functions:
            return cls._reward_functions[reward_name]()
        return None

    @classmethod
    def get_available_rewards(cls) -> List[str]:
        """获取所有可用的奖励函数名称"""
        return list(cls._reward_functions.keys())


# 注册奖励函数
RewardFunctionFactory.register("box_iou", BoxIoURewardFunction)
RewardFunctionFactory.register("point_in_box", PointInBoxRewardFunction)
RewardFunctionFactory.register("point_in_mask", PointInMaskRewardFunction)
RewardFunctionFactory.register("mosaic_vqa", MosaicVQARewardFunction)


def compute_score(
    predicts: List[str], ground_truths: List[str]
) -> List[Dict[str, float]]:
    """批量计算分数 - 兼容性函数"""
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        # 处理Qwen2.5VL-32B格式
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)

        reward_name = ground_truth["reward_name"]
        # 获取对应的奖励函数
        reward_func = RewardFunctionFactory.get_reward_function(reward_name)
        if reward_func is None:
            print(f"Warning: Unknown reward function '{reward_name}', using default")
            reward_func = RewardFunctionFactory.get_reward_function("box_iou")

        accuracy_score = reward_func.compute_reward(predict, ground_truth)
        scores.append(
            {
                "overall": accuracy_score,
                f"{reward_name}": accuracy_score,
            }
        )

    return scores


# 兼容性函数 - 保持向后兼容
def accuracy_reward(predict: str, ground_truth: str) -> float:
    """兼容性函数，使用默认的box_iou奖励"""
    reward_func = RewardFunctionFactory.get_reward_function("box_iou")
    return reward_func.compute_reward(predict, ground_truth)


def ensure_dir_exists(path: str) -> bool:
    """确保目录存在"""
    try:
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        print(f"Failed to create directory: {e}")
        return False


def create_visualization(
    gt_data: Dict,
    pred_data: Dict,
    reward_score: float,
    reward_name: str,
    dataset_name: str,
    image_data: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    创建GT和预测结果的可视化对比图

    Args:
        gt_data: Ground truth数据
        pred_data: 预测数据
        reward_score: 奖励分数
        reward_name: 奖励函数名称
        dataset_name: 数据集名称
        image_data: Base64编码的图像数据（可选）
        save_path: 保存路径（可选）
    """
    try:
        # 获取图像尺寸
        width, height = gt_data.get("dims", (1000, 1000))

        # 创建子图
        fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(20, 10))

        # 如果有图像数据，解码并显示
        if image_data:
            try:
                # 解码base64图像
                img_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img = img.resize((width, height))

                ax_gt.imshow(img)
                ax_pred.imshow(img)
            except Exception as e:
                print(f"Failed to decode image: {e}")
                # 创建空白背景
                ax_gt.set_xlim(0, width)
                ax_gt.set_ylim(height, 0)
                ax_pred.set_xlim(0, width)
                ax_pred.set_ylim(height, 0)
        else:
            # 创建空白背景
            ax_gt.set_xlim(0, width)
            ax_gt.set_ylim(height, 0)
            ax_pred.set_xlim(0, width)
            ax_pred.set_ylim(height, 0)

        # 设置标题
        ax_gt.set_title(f"Ground Truth\n{dataset_name}", fontsize=14, fontweight="bold")
        ax_pred.set_title(
            f"Prediction (Reward: {reward_score:.3f})\n{reward_name}",
            fontsize=14,
            fontweight="bold",
        )

        # 根据奖励类型进行可视化
        if reward_name == "box_iou":
            _visualize_box_iou(ax_gt, ax_pred, gt_data, pred_data)
        elif reward_name == "point_in_box":
            _visualize_point_in_box(ax_gt, ax_pred, gt_data, pred_data)
        elif reward_name == "point_in_mask":
            _visualize_point_in_mask(ax_gt, ax_pred, gt_data, pred_data, width, height)
        elif reward_name == "rejection":
            _visualize_rejection(ax_gt, ax_pred, gt_data, pred_data)

        ax_gt.axis("off")
        ax_pred.axis("off")
        plt.tight_layout()

        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
            plt.close()
        else:
            plt.show()

    except Exception as e:
        print(f"Visualization failed: {e}")
        if "fig" in locals():
            plt.close(fig)


def _visualize_box_iou(ax_gt, ax_pred, gt_data, pred_data):
    """可视化Box IoU任务"""
    # 获取颜色映射
    gt_objects = gt_data.get("objects", {})
    pred_objects = pred_data.get("objects", {})

    all_categories = set(gt_objects.keys()) | set(pred_objects.keys())
    colors = plt.cm.tab20(range(len(all_categories)))
    category_to_color = {cat: colors[i] for i, cat in enumerate(all_categories)}

    # 绘制GT框
    for category, boxes in gt_objects.items():
        color = category_to_color[category]
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            rect = patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=3,
                edgecolor=color,
                facecolor=color,
                alpha=0.3,
            )
            ax_gt.add_patch(rect)

            # 添加标签
            label = f"{category}_{i+1}" if len(boxes) > 1 else category
            ax_gt.text(
                x0,
                y0 - 10,
                label,
                fontsize=10,
                color="white",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
            )

    # 绘制预测框
    for category, boxes in pred_objects.items():
        color = category_to_color[category]
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            rect = patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=3,
                edgecolor=color,
                facecolor=color,
                alpha=0.3,
            )
            ax_pred.add_patch(rect)

            # 添加标签
            label = f"{category}_{i+1}" if len(boxes) > 1 else category
            ax_pred.text(
                x0,
                y0 - 10,
                label,
                fontsize=10,
                color="white",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
            )


def _visualize_point_in_box(ax_gt, ax_pred, gt_data, pred_data):
    """可视化Point in Box任务"""
    # GT: 显示框和点
    gt_objects = gt_data.get("objects", {})
    pred_objects = pred_data.get("objects", {})

    all_categories = set(gt_objects.keys()) | set(pred_objects.keys())
    colors = plt.cm.tab20(range(len(all_categories)))
    category_to_color = {cat: colors[i] for i, cat in enumerate(all_categories)}

    # 绘制GT（框和点）
    for category, obj_data in gt_objects.items():
        color = category_to_color[category]
        boxes = obj_data.get("boxes", [])
        points = obj_data.get("points", [])

        # 绘制框
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            rect = patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=3,
                edgecolor=color,
                facecolor="none",
                alpha=0.8,
            )
            ax_gt.add_patch(rect)

        # 绘制GT点
        for i, point in enumerate(points):
            x, y = point
            circle = patches.Circle(
                (x, y),
                radius=8,
                linewidth=3,
                edgecolor="white",
                facecolor=color,
                alpha=0.8,
            )
            ax_gt.add_patch(circle)

        # 添加标签
        if boxes and points:
            label = f"{category} (GT)"
            ax_gt.text(
                boxes[0][0],
                boxes[0][1] - 15,
                label,
                fontsize=10,
                color="white",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
            )

    # 绘制预测点
    for category, points in pred_objects.items():
        color = category_to_color[category]
        for i, point in enumerate(points):
            x, y = point
            circle = patches.Circle(
                (x, y),
                radius=8,
                linewidth=3,
                edgecolor="yellow",
                facecolor=color,
                alpha=0.8,
            )
            ax_pred.add_patch(circle)

            # 添加标签
            label = f"{category}_pred_{i+1}" if len(points) > 1 else f"{category}_pred"
            ax_pred.text(
                x + 15,
                y,
                label,
                fontsize=10,
                color="white",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
            )


def _visualize_point_in_mask(ax_gt, ax_pred, gt_data, pred_data, width, height):
    """可视化Point in Mask任务"""
    gt_objects = gt_data.get("objects", {})
    pred_objects = pred_data.get("objects", {})

    all_categories = set(gt_objects.keys()) | set(pred_objects.keys())
    colors = plt.cm.tab20(range(len(all_categories)))
    category_to_color = {cat: colors[i] for i, cat in enumerate(all_categories)}

    # 绘制GT（框、掩码和点）
    for category, obj_data in gt_objects.items():
        color = category_to_color[category]
        boxes = obj_data.get("boxes", [])
        masks = obj_data.get("masks", [])
        points = obj_data.get("points", [])

        # 绘制框
        for box in boxes:
            x0, y0, x1, y1 = box
            rect = patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=3,
                edgecolor=color,
                facecolor="none",
                alpha=0.8,
            )
            ax_gt.add_patch(rect)

        # 绘制掩码
        for mask in masks:
            try:
                if isinstance(mask, dict) and "counts" in mask:
                    binary_mask = coco_mask.decode(mask)
                elif isinstance(mask, list):
                    binary_mask = np.array(mask)
                else:
                    continue

                # 创建掩码轮廓
                contours = _get_mask_contours(binary_mask)
                for contour in contours:
                    polygon = patches.Polygon(
                        contour,
                        closed=True,
                        linewidth=2,
                        edgecolor=color,
                        facecolor=color,
                        alpha=0.3,
                    )
                    ax_gt.add_patch(polygon)
            except Exception as e:
                print(f"Failed to visualize mask: {e}")

        # 绘制GT点
        for point in points:
            x, y = point
            circle = patches.Circle(
                (x, y),
                radius=8,
                linewidth=3,
                edgecolor="white",
                facecolor=color,
                alpha=0.8,
            )
            ax_gt.add_patch(circle)

    # 绘制预测点
    for category, points in pred_objects.items():
        color = category_to_color[category]
        for i, point in enumerate(points):
            x, y = point
            circle = patches.Circle(
                (x, y),
                radius=8,
                linewidth=3,
                edgecolor="yellow",
                facecolor=color,
                alpha=0.8,
            )
            ax_pred.add_patch(circle)

            # 添加标签
            label = f"{category}_pred_{i+1}" if len(points) > 1 else f"{category}_pred"
            ax_pred.text(
                x + 15,
                y,
                label,
                fontsize=10,
                color="white",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
            )


def _visualize_rejection(ax_gt, ax_pred, gt_data, pred_data):
    """可视化Rejection任务"""
    # GT: 显示应该为空
    ax_gt.text(
        0.5,
        0.5,
        "Should be EMPTY\n(No objects)",
        transform=ax_gt.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        color="green",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
    )

    # 预测结果
    pred_objects = pred_data.get("objects", {})
    total_predictions = sum(len(coords) for coords in pred_objects.values())

    if total_predictions == 0:
        # 正确拒绝
        ax_pred.text(
            0.5,
            0.5,
            "CORRECT REJECTION\n(No predictions)",
            transform=ax_pred.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            color="green",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
        )
    else:
        # 产生幻觉
        ax_pred.text(
            0.5,
            0.5,
            f"HALLUCINATION\n({total_predictions} predictions)",
            transform=ax_pred.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            color="red",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8),
        )

        # 显示错误的预测
        colors = plt.cm.tab20(range(len(pred_objects)))
        color_idx = 0
        for category, coords_list in pred_objects.items():
            color = colors[color_idx % len(colors)]
            color_idx += 1

            for i, coords in enumerate(coords_list):
                if len(coords) == 4:  # 边界框
                    x0, y0, x1, y1 = coords
                    rect = patches.Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        linewidth=3,
                        edgecolor=color,
                        facecolor=color,
                        alpha=0.3,
                    )
                    ax_pred.add_patch(rect)
                elif len(coords) == 2:  # 点
                    x, y = coords
                    circle = patches.Circle(
                        (x, y),
                        radius=8,
                        linewidth=3,
                        edgecolor="red",
                        facecolor=color,
                        alpha=0.8,
                    )
                    ax_pred.add_patch(circle)


def _get_mask_contours(binary_mask):
    """从二进制掩码中提取轮廓"""
    try:
        import cv2

        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # 转换为matplotlib可用的格式
        result = []
        for contour in contours:
            if len(contour) > 2:
                points = contour.reshape(-1, 2)
                result.append(points)
        return result
    except ImportError:
        # 如果没有cv2，使用简单的边界
        y_coords, x_coords = np.where(binary_mask)
        if len(y_coords) > 0:
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            return [
                np.array(
                    [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
                )
            ]
        return []
