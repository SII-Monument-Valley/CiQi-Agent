#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索结果数据模型

包含文本搜索、图像搜索和CLIP文本搜索的结果数据类，
以及相关的图像加载功能。
"""

import os
from typing import List, Union, Dict, Any, Optional
from PIL import Image
from dataclasses import dataclass

# Optional local roots are configured with the platform path separator, for
# example: CIQI_AGENT_IMAGE_ROOTS=/data/raw:/data/resized
BACKUP_IMAGE_DIRS = [
    value
    for value in os.getenv(
        "CIQI_AGENT_IMAGE_ROOTS",
        os.pathsep.join(("./data/processed/museum/raw", "./data/processed/museum/resized", "./images")),
    ).split(os.pathsep)
    if value
]


def load_image_from_path(image_path: str, image_file: str = None, backup_dirs: List[str] = None) -> Optional[Image.Image]:
    """
    从文件系统加载图像，支持多种查找策略

    Args:
        image_path: 完整的图像路径
        image_file: 图像文件名（用于备用查找）
        backup_dirs: 备用目录列表

    Returns:
        PIL.Image对象，如果找不到则返回None
    """
    if backup_dirs is None:
        backup_dirs = BACKUP_IMAGE_DIRS

    # 策略1: 尝试直接使用image_path
    if image_path and os.path.exists(image_path):
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"无法加载图像 {image_path}: {e}")

    # 策略2: 如果image_path不存在，尝试从备用目录查找image_file
    if image_file:
        for backup_dir in backup_dirs:
            if os.path.exists(backup_dir):
                potential_path = os.path.join(backup_dir, image_file)
                if os.path.exists(potential_path):
                    try:
                        return Image.open(potential_path).convert("RGB")
                    except Exception as e:
                        print(f"无法加载图像 {potential_path}: {e}")
                        continue

        # 策略3: 在备用目录中递归查找image_file
        for backup_dir in backup_dirs:
            if os.path.exists(backup_dir):
                try:
                    for root, dirs, files in os.walk(backup_dir):
                        if image_file in files:
                            found_path = os.path.join(root, image_file)
                            try:
                                return Image.open(found_path).convert("RGB")
                            except Exception as e:
                                print(f"无法加载图像 {found_path}: {e}")
                                continue
                except Exception as e:
                    print(f"搜索目录 {backup_dir} 时出错: {e}")
                    continue

    print(f"未找到图像: {image_path} 或 {image_file}")
    return None


@dataclass
class BaseSearchResult:
    """搜索结果基类"""
    id: str
    score: float

    def __str__(self) -> str:
        """字符串表示"""
        return f"  [{self.__class__.__name__}] ID: {self.id}, Score: {self.score:.3f}"


@dataclass
class TextSearchResult(BaseSearchResult):
    """文本搜索结果数据类"""
    source: str
    caption: str
    text: str
    perplexity: float
    type: str = "long_text"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextSearchResult':
        """从字典创建TextSearchResult对象"""
        payload = data.get('payload', {})
        return cls(
            id=data.get('id', ''),
            score=data.get('score', 0.0),
            source=payload.get('source', ''),
            caption=payload.get('caption', ''),
            text=payload.get('text', ''),
            perplexity=payload.get('perplexity', 0.0),
            type=payload.get('type', 'long_text')
        )

    def __str__(self) -> str:
        """字符串表示"""
        return f"  [文本结果] 文本: {self.text[:100]}...\n  - 来源: {self.source}\n  - 分数: {self.score}\n  - 困惑度: {self.perplexity}"


@dataclass
class ImageSearchResult(BaseSearchResult):
    """图像搜索结果数据类"""
    uuid: str
    image_file: str
    image_path: str
    name: str
    description: str
    object_type: str
    color: str
    decoration: str
    dynasty: str
    reign: str
    source: str
    image_index: int
    type: str = "image"
    _loaded_image: Optional[Image.Image] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImageSearchResult':
        """从字典创建ImageSearchResult对象"""
        payload = data.get('payload', {})
        instance = cls(
            id=data.get('id', ''),
            score=data.get('score', 0.0),
            uuid=payload.get('uuid', ''),
            image_file=payload.get('image_file', ''),
            image_path=payload.get('image_path', ''),
            name=payload.get('name', ''),
            description=payload.get('description', ''),
            object_type=payload.get('object_type', ''),
            color=payload.get('color', ''),
            decoration=payload.get('decoration', ''),
            dynasty=payload.get('dynasty', ''),
            reign=payload.get('reign', ''),
            source=payload.get('source', ''),
            image_index=payload.get('image_index', 0),
            type=payload.get('type', 'image')
        )

        # 自动加载图像
        instance._loaded_image = load_image_from_path(instance.image_path, instance.image_file)
        return instance

    def load_image(self) -> Optional[Image.Image]:
        """获取已加载的图像"""
        return self._loaded_image

    def reload_image(self) -> Optional[Image.Image]:
        """重新加载图像"""
        self._loaded_image = load_image_from_path(self.image_path, self.image_file)
        return self._loaded_image

    def __str__(self) -> str:
        """字符串表示"""
        dynasty_reign = f"{self.dynasty} {self.reign}".strip()
        result = f"  [图像结果] 名称: {self.name}\n  - 器型: {self.object_type}\n  - 颜色: {self.color}\n  - 装饰: {self.decoration}\n  - 朝代: {dynasty_reign}\n  - 来源: {self.source}\n  - 分数: {self.score}\n  - 图像文件: {self.image_file}"

        # 尝试加载图像信息
        image = self.load_image()
        if image:
            result += f"\n  - 图像已加载: {image.size} ({image.mode})"
        else:
            result += "\n  - 图像加载失败"

        return result


@dataclass
class ClipTextSearchResult(BaseSearchResult):
    """CLIP文本搜索结果数据类（包含图像信息的文本结果）"""
    uuid: str
    name: str
    caption: str
    image_file: str
    image_path: str
    description: str
    object_type: str
    color: str
    decoration: str
    dynasty: str
    reign: str
    source: str
    type: str = "clip_text"
    _loaded_image: Optional[Image.Image] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClipTextSearchResult':
        """从字典创建ClipTextSearchResult对象"""
        payload = data.get('payload', {})
        instance = cls(
            id=data.get('id', ''),
            score=data.get('score', 0.0),
            uuid=payload.get('uuid', ''),
            name=payload.get('name', ''),
            caption=payload.get('caption', ''),
            image_file=payload.get('image_file', ''),
            image_path=payload.get('image_path', ''),
            description=payload.get('description', ''),
            object_type=payload.get('object_type', ''),
            color=payload.get('color', ''),
            decoration=payload.get('decoration', ''),
            dynasty=payload.get('dynasty', ''),
            reign=payload.get('reign', ''),
            source=payload.get('source', ''),
            type=payload.get('type', 'clip_text')
        )

        # 自动加载图像
        instance._loaded_image = load_image_from_path(instance.image_path, instance.image_file)
        return instance

    def load_image(self) -> Optional[Image.Image]:
        """获取已加载的图像"""
        return self._loaded_image

    def reload_image(self) -> Optional[Image.Image]:
        """重新加载图像"""
        self._loaded_image = load_image_from_path(self.image_path, self.image_file)
        return self._loaded_image

    def __str__(self) -> str:
        """字符串表示"""
        dynasty_reign = f"{self.dynasty} {self.reign}".strip()
        result = f"  [CLIP文本结果] 名称: {self.name}\n  - 标题: {self.caption}\n  - 器型: {self.object_type}\n  - 颜色: {self.color}\n  - 装饰: {self.decoration}\n  - 朝代: {dynasty_reign}\n  - 来源: {self.source}\n  - 分数: {self.score}\n  - 图像文件: {self.image_file}"

        # 尝试加载图像信息
        image = self.load_image()
        if image:
            result += f"\n  - 图像已加载: {image.size} ({image.mode})"
        else:
            result += "\n  - 图像加载失败"

        return result


@dataclass
class SearchResponse:
    """搜索结果响应数据类"""
    hits: List[Union[TextSearchResult, ImageSearchResult, ClipTextSearchResult]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any], result_type: str = "text") -> 'SearchResponse':
        """从字典创建SearchResponse对象"""
        hits_data = data.get('hits', [])
        hits = []

        for hit_data in hits_data:
            payload = hit_data.get('payload', {})
            hit_type = payload.get('type', result_type)

            if hit_type == 'image':
                hits.append(ImageSearchResult.from_dict(hit_data))
            elif hit_type == 'clip_text':
                hits.append(ClipTextSearchResult.from_dict(hit_data))
            else:
                hits.append(TextSearchResult.from_dict(hit_data))

        return cls(hits=hits)

    def __str__(self) -> str:
        """字符串表示"""
        result = ""
        for i, hit in enumerate(self.hits):
            result += f"结果 {i+1}:\n{hit}\n\n"
        return result.rstrip()


def load_images_from_results(results: List[SearchResponse]) -> Dict[str, Image.Image]:
    """
    从搜索结果中批量加载图像

    Args:
        results: 搜索结果列表

    Returns:
        字典，键为结果ID，值为PIL.Image对象
    """
    loaded_images = {}

    for result in results:
        for hit in result.hits:
            # 检查是否是包含图像信息的结果
            if hasattr(hit, 'image_path') and hasattr(hit, 'image_file'):
                image = hit.load_image()
                if image:
                    loaded_images[hit.id] = image

    return loaded_images


def get_image_by_id(results: List[SearchResponse], result_id: str) -> Optional[Image.Image]:
    """
    根据结果ID获取图像

    Args:
        results: 搜索结果列表
        result_id: 结果ID

    Returns:
        PIL.Image对象，如果找不到则返回None
    """
    for result in results:
        for hit in result.hits:
            if hit.id == result_id and hasattr(hit, 'load_image'):
                return hit.load_image()
    return None


def get_results_by_type(results: List[SearchResponse], result_type: str) -> List[Union[TextSearchResult, ImageSearchResult, ClipTextSearchResult]]:
    """
    根据类型筛选搜索结果

    Args:
        results: 搜索结果列表
        result_type: 结果类型 ('text', 'image', 'clip_text')

    Returns:
        筛选后的结果列表
    """
    filtered_results = []

    for result in results:
        for hit in result.hits:
            if hit.type == result_type:
                filtered_results.append(hit)

    return filtered_results


def print_search_results(results: List[SearchResponse], show_images: bool = True):
    """
    格式化打印搜索结果

    Args:
        results: 搜索结果列表
        show_images: 是否显示图像信息（已集成到__str__方法中）
    """
    for i, result in enumerate(results):
        print(f"查询 {i+1} 结果:")
        print(result)


# 导出所有类和函数
__all__ = [
    'BaseSearchResult',
    'TextSearchResult',
    'ImageSearchResult',
    'ClipTextSearchResult',
    'SearchResponse',
    'load_image_from_path',
    'load_images_from_results',
    'get_image_by_id',
    'get_results_by_type',
    'print_search_results',
    'BACKUP_IMAGE_DIRS'
]
