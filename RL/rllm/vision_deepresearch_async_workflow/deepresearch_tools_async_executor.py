from vision_deepresearch_async_workflow.tools.python_interpreter_tool import (
    PythonInterpreterTool,
)
from vision_deepresearch_async_workflow.tools.shared import DeepResearchTool
from vision_deepresearch_async_workflow.tools.visual_tools import (
    CropTool,
    ImageSearchTool,
    LayoutParsingTool,
    PerspectiveCorrectTool,
    SharpenTool,
    SuperResolutionTool,
    TextSearchTool,
    WebSearchTool,
)

DEEPRESEARCH_TOOLS = {
    "crop": CropTool(),
    "layout_parsing": LayoutParsingTool(),
    "text_search": TextSearchTool(),
    "image_search": ImageSearchTool(),
    "web_search": WebSearchTool(),
    "perspective_correct": PerspectiveCorrectTool(),
    "super_resolution": SuperResolutionTool(),
    "sharpen": SharpenTool(),
    "PythonInterpreter": PythonInterpreterTool(),
}


def get_tool(name: str) -> DeepResearchTool:
    """Get a tool by name."""
    return DEEPRESEARCH_TOOLS.get(name)


def get_all_tools() -> dict[str, DeepResearchTool]:
    """Get all available tools."""
    return DEEPRESEARCH_TOOLS.copy()


__all__ = [
    "DeepResearchTool",
    "CropTool",
    "LayoutParsingTool",
    "TextSearchTool",
    "ImageSearchTool",
    "WebSearchTool",
    "PerspectiveCorrectTool",
    "SuperResolutionTool",
    "SharpenTool",
    "PythonInterpreterTool",
    "DEEPRESEARCH_TOOLS",
    "get_tool",
    "get_all_tools",
]
