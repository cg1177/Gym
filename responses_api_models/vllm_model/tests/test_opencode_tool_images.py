# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from typing import Any, Dict, List
from unittest.mock import MagicMock

from pytest import raises

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymChatCompletionCreateParamsNonStreaming
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig
from responses_api_models.vllm_model.opencode_tool_images import (
    ATTACHED_MEDIA_TEXT,
    has_attached_media,
    inline_opencode_tool_images,
)


# The text of a watch_video result as the video tools write it (resources_servers/video_mcqa/sandbox_image).
WATCH_HEADER = (
    "Video: videos/clip.mp4\n"
    "Source range: [0.00s, 2.00s)\n"
    "Temporal labels are sampling times; original source PTS is preserved separately.\n"
    "Frames: 4 total, 4 unique source frames (0.0% repeated)\n"
    "Resolution: 224×112\n"
    "Visual tokens: 128\n"
    "Sampling interval: 0.5 s\n"
    "Maximum gap between distinct source-frame PTS: 0.5005 s"
)


def watch_text(*times: float) -> str:
    frames = "\n".join(f"Frame {number} sampled at {time:.2f} seconds:" for number, time in enumerate(times, 1))
    return f"{WATCH_HEADER}\nThis is a video:\n{frames}"


def image(name: str) -> Dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{name}"}}


def attached(*images: Dict[str, Any]) -> Dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": ATTACHED_MEDIA_TEXT}, *images]}


def assistant(*calls: tuple, text: str = "") -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": text,
        "reasoning_content": "Look at the video.",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}
            for call_id, name in calls
        ],
    }


def tool(call_id: str, text: str) -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def frame_part(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


PROMPT = [{"role": "system", "content": "You are opencode."}, {"role": "user", "content": "Watch the video."}]


def test_watch_video_frames_follow_their_labels_as_in_the_training_data() -> None:
    messages = [
        *PROMPT,
        assistant(("c1", "watch_video")),
        tool("c1", watch_text(0.25, 0.75)),
        attached(image("f1"), image("f2")),
    ]
    inlined = inline_opencode_tool_images(messages)
    assert inlined[:3] == messages[:3] and len(inlined) == 4
    assert inlined[3] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": [
            frame_part(WATCH_HEADER),
            frame_part("\nThis is a video:\nFrame 1 sampled at 0.25 seconds: "),
            image("f1"),
            frame_part("\nFrame 2 sampled at 0.75 seconds: "),
            image("f2"),
        ],
    }
    # The text is the tool's own, with a space between each label and its frame.
    text = "".join(part["text"] for part in inlined[3]["content"] if part["type"] == "text")
    assert text == watch_text(0.25, 0.75).replace("seconds:", "seconds: ")


def test_one_response_s_images_are_split_by_tool_in_order() -> None:
    bash = tool("c2", "duration=140.1")
    messages = [
        *PROMPT,
        assistant(("c1", "watch_video"), ("c2", "bash"), ("c3", "view_image"), ("c4", "read"), ("c5", "watch_video")),
        tool("c1", watch_text(0.25, 0.75)),
        bash,
        tool("c3", "Image: work/fr/f_11.jpg"),
        tool("c4", "Image read successfully"),
        tool("c5", watch_text(1.25)),
        attached(image("a"), image("b"), image("view"), image("read"), image("c")),
    ]
    inlined = inline_opencode_tool_images(messages)
    assert [message["role"] for message in inlined] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "tool",
        "tool",
        "tool",
    ]
    assert [part for part in inlined[3]["content"] if part["type"] == "image_url"] == [image("a"), image("b")]
    assert inlined[4] is bash
    assert inlined[5]["content"] == [frame_part("Image: work/fr/f_11.jpg"), image("view")]
    assert inlined[6]["content"] == [frame_part("Image read successfully"), image("read")]
    assert inlined[7]["content"][-2:] == [
        frame_part("\nThis is a video:\nFrame 1 sampled at 1.25 seconds: "),
        image("c"),
    ]
    assert not has_attached_media(inlined)


def test_a_result_without_images_takes_none() -> None:
    # The video tools' media budget replaces a result's frames with a notice; a failed call has no frames.
    notice = "Tool result images not shown: the episode's tool images passed max_tool_media_bytes."
    messages = [
        *PROMPT,
        assistant(("c1", "watch_video"), ("c2", "watch_video"), ("c3", "view_image")),
        tool("c1", notice),
        tool("c2", "ValueError: source frame timestamps must be strictly increasing"),
        tool("c3", "Image: videos/iris_left.png"),
        attached(image("view")),
    ]
    inlined = inline_opencode_tool_images(messages)
    assert inlined[3:5] == messages[3:5]
    assert inlined[5]["content"] == [frame_part("Image: videos/iris_left.png"), image("view")]
    assert len(inlined) == 6


def test_responses_whose_images_cannot_be_placed_stay_as_opencode_sent_them() -> None:
    cases = [
        # One image more than the results announce.
        [assistant(("c1", "view_image")), tool("c1", "Image: a.png"), attached(image("a"), image("extra"))],
        # A tool these rules do not know returns an image.
        [assistant(("c1", "screenshot")), tool("c1", "Saved the screen"), attached(image("s"))],
        # A part that is not an image, e.g. the PDF that OpenCode's read returns.
        [
            assistant(("c1", "read")),
            tool("c1", "PDF read successfully"),
            {"role": "user", "content": [{"type": "text", "text": ATTACHED_MEDIA_TEXT}, {"type": "file"}]},
        ],
        # Frame lines apart from the video prompt.
        [
            assistant(("c1", "watch_video")),
            tool("c1", "This is a video:\nnote\nFrame 1 sampled at 0.25 seconds:"),
            attached(image("f")),
        ],
        # A tool message whose content is not a string.
        [
            assistant(("c1", "view_image")),
            {**tool("c1", ""), "content": [frame_part("Image: a.png")]},
            attached(image("a")),
        ],
    ]
    for group in cases:
        messages = [*PROMPT, *group]
        assert inline_opencode_tool_images(messages) == messages
        assert has_attached_media(inline_opencode_tool_images(messages))


def test_other_messages_are_kept_as_they_are_and_the_input_is_not_changed() -> None:
    # No attached-media message (tools without images), an attached-media message after a tool that returns no
    # image (kept), a final answer.
    lookalike = attached(image("u"))
    messages = [
        *PROMPT,
        assistant(("c1", "bash")),
        tool("c1", "ok"),
        lookalike,
        assistant(("c2", "view_image")),
        tool("c2", "Image: a.png"),
        attached(image("a")),
        {"role": "assistant", "content": "\\boxed{B}"},
    ]
    before = deepcopy(messages)
    inlined = inline_opencode_tool_images(messages)
    assert messages == before
    assert all(new is old for new, old in zip(inlined[:6], messages[:6]))
    assert inlined[6]["content"] == [frame_part("Image: a.png"), image("a")] and inlined[7] is messages[8]
    assert inline_opencode_tool_images(PROMPT) == PROMPT


def opencode_episode() -> List[List[Dict[str, Any]]]:
    """The requests of one OpenCode episode: each call sends the whole history again, with the newest tool results."""
    turns = [
        (assistant(("c1", "bash")), [tool("c1", "duration=2.0")], None),
        (assistant(("c2", "watch_video")), [tool("c2", watch_text(0.25, 0.75))], attached(image("f1"), image("f2"))),
        (
            assistant(("c3", "watch_video"), ("c4", "bash"), ("c5", "view_image")),
            [
                tool("c3", watch_text(1.25)),
                tool("c4", "made /tmp/opencode/x.jpg"),
                tool("c5", "Image: /tmp/opencode/x.jpg"),
            ],
            attached(image("f3"), image("x")),
        ),
        # Left as sent: its image count does not add up.
        (assistant(("c6", "screenshot")), [tool("c6", "Saved the screen")], attached(image("s"))),
        (assistant(("c7", "read")), [tool("c7", "Image read successfully")], attached(image("r"))),
    ]
    requests, history = [], list(PROMPT)
    for reply, results, media in turns:
        requests.append(deepcopy(history))
        history += [reply, *results, *([media] if media else [])]
    requests.append(deepcopy(history))
    return requests


def render(messages: List[Dict[str, Any]]) -> str:
    """A flat rendering in message and part order, as chat templates place text and image placeholders."""
    out = []
    for message in messages:
        content = message.get("content")
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content or []
        body = "".join(part["text"] if part["type"] == "text" else "<image>" for part in parts)
        calls = "".join(f"<call {call['id']}>" for call in message.get("tool_calls") or [])
        out.append(f"<{message['role']}>{body}{calls}</{message['role']}>")
    return "".join(out)


def test_the_rewrite_is_prefix_stable_across_opencode_s_repeated_history() -> None:
    requests = opencode_episode()
    inlined = [inline_opencode_tool_images(request) for request in requests]
    for number in range(len(requests) - 1):
        earlier, later = inlined[number], inlined[number + 1]
        reply = requests[number + 1][len(requests[number])]
        # Each call's prompt is the previous call's prompt, then the reply it got, then the new tool results.
        assert later[: len(earlier)] == earlier
        assert later[len(earlier)] == reply
        assert render(later).startswith(render(earlier) + render([reply]))
        # The same request rewritten again gives the same messages.
        assert inline_opencode_tool_images(deepcopy(requests[number])) == earlier
    final = inlined[-1]
    # Every response whose images add up has them inside its results; the screenshot's stays as OpenCode sent it.
    assert [message for message in final if message["role"] == "user"][1:] == [attached(image("s"))]
    images = [
        part["image_url"]["url"]
        for message in final
        for part in message["content"]
        if isinstance(message["content"], list) and part["type"] == "image_url"
    ]
    assert images == [f"data:image/jpeg;base64,{name}" for name in ("f1", "f2", "f3", "x", "s", "r")]


def model(**overrides: Any) -> VLLMModel:
    config = VLLMModelConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="policy_model",
        base_url="http://localhost:9999/v1",
        api_key="dummy_key",  # pragma: allowlist secret
        model="dummy-model",
        return_token_id_information=False,
        uses_reasoning_parser=False,
        uses_interleaved_reasoning=False,
        **overrides,
    )
    return VLLMModel(config=config, server_client=MagicMock(spec=ServerClient))


def test_the_model_server_moves_opencode_s_tool_images_only_when_asked() -> None:
    messages = [*PROMPT, assistant(("c1", "watch_video")), tool("c1", watch_text(0.25)), attached(image("f1"))]
    # The request as Gym's /v1/chat/completions validates it.
    body = NeMoGymChatCompletionCreateParamsNonStreaming(model="dummy_model", messages=messages)
    sent = model(inline_opencode_tool_images=True)._preprocess_chat_completion_create_params(
        MagicMock(), body.model_dump(exclude_unset=True)
    )
    assert [message["role"] for message in sent["messages"]] == ["system", "user", "assistant", "tool"]
    assert sent["messages"][3]["content"][-1]["image_url"]["url"] == "data:image/jpeg;base64,f1"
    unchanged = model()._preprocess_chat_completion_create_params(MagicMock(), body.model_dump(exclude_unset=True))
    assert [message["role"] for message in unchanged["messages"]] == ["system", "user", "assistant", "tool", "user"]


def test_the_move_needs_the_chat_completions_endpoint() -> None:
    with raises(ValueError, match="inline_opencode_tool_images requires use_completions_api=false"):
        model(inline_opencode_tool_images=True, use_completions_api=True)
