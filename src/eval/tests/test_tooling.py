from __future__ import annotations

import pytest
from ciqi_eval.tooling import ToolRegistry, zoom_image
from ciqi_eval.types import ToolContext
from PIL import Image


def test_zoom_tool_maps_model_coordinates_back_to_raw_image() -> None:
    image = Image.new("RGB", (200, 100), "white")
    output = zoom_image(
        [image],
        [(2.0, 2.0)],
        index=1,
        bbox_2d=[10, 10, 40, 30],
        label="纹饰",
        min_dimension=10,
    )
    assert output.images[0].size == (60, 40)
    assert output.metadata["resolved_bbox"] == [20, 20, 80, 60]


@pytest.mark.asyncio
async def test_registry_rejects_unknown_tool() -> None:
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="Unsupported tool"):
        await registry.execute("missing", ToolContext(images=[], ratios=[]), {})
