# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode's tool images back inside the tool results they came from (``inline_opencode_tool_images``).

For an OpenAI-compatible provider, OpenCode 1.17.11 keeps the text of each tool result in its ``tool`` message and
sends the images of all the results of one model response in one ``user`` message after them: a text part
``Attached media from tool result:``, then the images in the order of the results. The model sees the labels and the
images apart and has to match them by order.

``inline_opencode_tool_images`` returns the messages with each such group rewritten as Nemotron 4's agentic video
training data has it: the attached-media message is gone and each tool message carries its own images, each after the
text that announces it.

- watch_video (resources_servers/video_mcqa's video tools): its text up to ``This is a video:``, then each frame's
  line ``Frame i sampled at t seconds:`` directly followed by a space and the frame.
- view_image (text ``Image: <path>``) and OpenCode's read of an image (``Image read successfully``): the text, then
  the image.
- A result whose images the video tools' media budget replaced with a notice (``Tool result images not shown: ...``),
  and every other tool: no image.

A group whose images do not add up to those counts, whose attached-media message holds a part that is not an image, or
whose tool message content is not a string is left as OpenCode sent it. Each group is rewritten from its own messages
only, so the history that OpenCode sends again with every call is rewritten the same way each time, and a call's
prompt still extends the previous call's prompt and reply.
"""

import re
from typing import Any, Dict, List, Optional


ATTACHED_MEDIA_TEXT = "Attached media from tool result:"
VIDEO_PROMPT = "This is a video:"
FRAME_LINE = re.compile(r"Frame \d+ sampled at \d+\.\d\d seconds:")
MEDIA_NOTICE = "Tool result images not shown:"
READ_MEDIA_TEXTS = ("Image read successfully", "PDF read successfully")


def tool_image_count(tool_name: Optional[str], text: str) -> int:
    """How many of the attached images the result of ``tool_name`` with ``text`` added."""
    if text.startswith(MEDIA_NOTICE):
        return 0
    if tool_name == "watch_video":
        return sum(1 for line in text.split("\n") if FRAME_LINE.fullmatch(line))
    if tool_name == "view_image":
        return int(text.startswith("Image: "))
    if tool_name == "read":
        return int(text in READ_MEDIA_TEXTS)
    return 0


def tool_content_with_images(
    tool_name: Optional[str], text: str, images: List[Dict[str, Any]]
) -> Optional[List[Dict[str, Any]]]:
    """The content parts of a tool result with its ``images`` in place, or None if they cannot be placed."""
    lines = text.split("\n")
    if tool_name != "watch_video" or VIDEO_PROMPT not in lines:
        return [{"type": "text", "text": text}, *images]
    start = lines.index(VIDEO_PROMPT)
    frames = [index for index, line in enumerate(lines) if FRAME_LINE.fullmatch(line)]
    if len(frames) != len(images) or frames != list(range(start + 1, start + 1 + len(frames))):
        return None
    parts = [{"type": "text", "text": "\n".join(lines[:start])}] if start else []
    for number, (index, image) in enumerate(zip(frames, images)):
        prefix = f"\n{VIDEO_PROMPT}\n" if number == 0 else "\n"
        parts += [{"type": "text", "text": f"{prefix}{lines[index]} "}, image]
    if rest := "\n".join(lines[frames[-1] + 1 :]).strip("\n"):
        parts.append({"type": "text", "text": "\n" + rest})
    return parts


def _attached_images(message: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """The parts after the text part of an attached-media message; None for any other message."""
    content = message.get("content")
    if message.get("role") != "user" or not isinstance(content, list) or not content:
        return None
    first = content[0]
    if not (isinstance(first, dict) and first.get("type") == "text" and first.get("text") == ATTACHED_MEDIA_TEXT):
        return None
    return content[1:]


def _inline_group(
    tool_names: Dict[Any, Optional[str]], results: List[Dict[str, Any]], attached: Dict[str, Any]
) -> Optional[List[Dict[str, Any]]]:
    """The tool messages of one model response with the images of ``attached`` inside them, or None to keep them."""
    images = _attached_images(attached)
    if images is None or not all(isinstance(part, dict) and part.get("type") == "image_url" for part in images):
        return None
    shares = []
    for result in results:
        # OpenCode sends a tool result's text as a string.
        text = result.get("content")
        if not isinstance(text, str):
            return None
        name = tool_names.get(result.get("tool_call_id"))
        shares.append((name, text, tool_image_count(name, text)))
    if sum(count for _, _, count in shares) != len(images):
        return None
    inlined = []
    for result, (name, text, count) in zip(results, shares):
        if not count:
            inlined.append(result)
            continue
        content = tool_content_with_images(name, text, images[:count])
        if content is None:
            return None
        images = images[count:]
        inlined.append({**result, "content": content})
    return inlined


def inline_opencode_tool_images(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``messages`` with each attached-media message's images moved into the tool messages before it (module docs).

    The input is not changed; messages that are not rewritten are the same objects in the result.
    """
    inlined: List[Dict[str, Any]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        inlined.append(message)
        index += 1
        calls = message.get("tool_calls") if isinstance(message, dict) and message.get("role") == "assistant" else None
        if not calls:
            continue
        end = index
        while end < len(messages) and isinstance(messages[end], dict) and messages[end].get("role") == "tool":
            end += 1
        if end == index or end == len(messages) or not isinstance(messages[end], dict):
            continue
        tool_names = {
            call.get("id"): (call.get("function") or {}).get("name") for call in calls if isinstance(call, dict)
        }
        group = _inline_group(tool_names, messages[index:end], messages[end])
        if group is None:
            continue
        inlined += group
        index = end + 1
    return inlined


def has_attached_media(messages: List[Dict[str, Any]]) -> bool:
    """Whether any of ``messages`` is an OpenCode attached-media message."""
    return any(isinstance(message, dict) and _attached_images(message) is not None for message in messages)
