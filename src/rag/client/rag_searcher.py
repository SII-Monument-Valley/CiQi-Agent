import os
import requests
from typing import List, Union, Dict, Any, Optional
from PIL import Image
import io
import json
from pathlib import Path

# 导入数据模型
from .search_results import (
    SearchResponse,
    load_images_from_results,
)


class RagSearcher:
    """RAG搜索器类，封装文本和图像搜索功能"""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 60, api_key: str = None):
        """
        初始化RAG搜索器

        Args:
            base_url: 服务端URL
            timeout: 请求超时时间（秒）
            api_key: API密钥（可选）
        """
        self.base_url = base_url
        self.timeout = timeout
        self.api_key = api_key

    def _headers(self) -> dict:
        """构建请求头，支持 API 鉴权"""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """封装 POST 请求发送 JSON 数据"""
        url = f"{self.base_url}{path}"
        response = requests.post(url, headers=self._headers(), data=json.dumps(payload), timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def convert_to_pil(image: Union[str, Image.Image, bytes]) -> Image.Image:
        """
        将图像输入转换为 PIL 图像对象。
        支持路径、PIL 图像对象、或图像二进制流（bytes）。
        """
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, bytes):
            try:
                return Image.open(io.BytesIO(image)).convert("RGB")
            except Exception as e:
                raise ValueError("Invalid image binary data")
        if isinstance(image, str):
            try:
                return Image.open(image).convert("RGB")
            except Exception as e:
                raise ValueError(f"Invalid image path: {image}")
        raise ValueError("Unsupported image input type")

    def search_text(self, texts: str | List[str], k: int = 10, alpha: float = 0.6) -> List[SearchResponse]:
        """
        批量发送文本查询到服务端（直接发送原始文本，服务端编码并返回结果）。

        Args:
            texts: 文本查询列表
            k: 返回结果数量
            alpha: 混合搜索权重

        Returns:
            每个查询一组命中结果（List[SearchResponse]）
        """
        if not isinstance(texts, list):
            texts = [texts]

        data = self._post_json("/search/text", {"texts": texts, "topk": k, "alpha": alpha})
        results = []
        for result_data in data["results"]:
            results.append(SearchResponse.from_dict(result_data, result_type="text"))
        return results

    def search_image(self, files: Union[str, Image.Image, bytes] | List[Union[str, Image.Image, bytes]], k: int = 10) -> List[SearchResponse]:
        """
        通过上传图像进行查询，支持路径、PIL 图像、二进制流等。

        Args:
            files: 图像文件列表，支持路径、PIL图像对象或二进制流
            k: 返回结果数量

        Returns:
            每个查询一组命中结果（List[SearchResponse]）
        """
        if not isinstance(files, list):
            files = [files]

        pil_images = [self.convert_to_pil(img) for img in files]  # 将图像转为 PIL 格式

        # 批量上传图片（通过文件上传）
        files_data = []
        for i, img in enumerate(pil_images):
            byte_io = io.BytesIO()
            img.save(byte_io, format="JPEG")
            byte_io.seek(0)
            files_data.append(("files", (f"image_{i}.jpg", byte_io, "image/jpeg")))

        try:
            # 发送请求
            response = requests.post(
                f"{self.base_url}/search/image",
                files=files_data,
                data={"topk": k},
                headers=self._headers(),
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            # 解析结果
            results = []
            for result_data in data["results"]:
                results.append(SearchResponse.from_dict(result_data, result_type="image"))
            return results

        except requests.HTTPError as e:
            if e.response and e.response.status_code == 404:
                print("Error: The batch image search endpoint is not available.")
            raise e

    def load_images_from_results(self, results: List[SearchResponse]) -> Dict[str, Image.Image]:
        """
        从搜索结果中批量加载图像

        Args:
            results: 搜索结果列表

        Returns:
            图像ID到PIL图像对象的映射
        """
        return load_images_from_results(results)


# -------------------- 3. 示例 --------------------
if __name__ == "__main__":
    import os

    # 创建RAG搜索器实例
    searcher = RagSearcher(
        base_url=os.environ.get("CIQI_AGENT_RAG_API_URL", "http://127.0.0.1:8000"),
        timeout=60,
        api_key=os.environ.get("CIQI_AGENT_RAG_API_KEY", ""),
    )

    print("=== 使用RagSearcher类进行搜索 ===")

    # 1. 文本查询示例
    print("\n--- 文本查询示例 ---")
    texts = ["青花龙纹罐"]
    text_results = searcher.search_text(texts, k=100, alpha=0.8)

    for i, result in enumerate(text_results):
        print(f"查询 {i+1} 结果:")
        print(result.hits[0])

    # # 3. 批量加载所有图像
    # print("\n--- 批量加载图像 ---")
    # loaded_images = searcher.load_images_from_results(text_results)
    # print(f"成功加载 {len(loaded_images)} 张图像")
    # for result_id, image in loaded_images.items():
    #     print(f"  - {result_id}: {image.size} ({image.mode})")

    # # 4. 图像查询示例
    # print("\n--- 图像查询示例 ---")
    # image_root_path = "./data/processed/museum/raw"
    # image_paths = [
    #     f"{image_root_path}/MUS_raw_9d606ac13d54342c6f7695ad99493b0301f1d64eed947be1fb6b91b5df1f2888.jpg"
    # ]
    # image_results = searcher.search_image(image_paths, k=10)

    # for i, result in enumerate(image_results):
    #     print(f"图像查询 {i+1} 结果:")
    #     print(result)

    # print("\n=== 查询完成 ===")
