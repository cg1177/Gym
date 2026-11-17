# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-turn rows for open-ended video questions: the model sees the video and the question, never the rubric."""

from collections.abc import Sequence
from typing import Any, TypedDict

from resources_servers.video_mcqa.task import ResponsesCreateParams, Setting, video_parts
from resources_servers.video_qa_rubric.task_data import validate_rubric


PROMPT = "Watch the video and answer the question.\n\nQuestion: {question}"


class PreparedTask(TypedDict):
    """A Gym row: the model's input in ``responses_create_params``, the grading data in ``verifier_metadata``."""

    task_id: str
    setting: Setting
    responses_create_params: ResponsesCreateParams
    verifier_metadata: dict[str, Any]


def prepare_single_turn(
    task_id: str,
    question: str,
    rubric: dict[str, Any],
    *,
    video_url: str | None = None,
    frames: Sequence[tuple[str, str]] | None = None,
) -> PreparedTask:
    """One user turn with the video, then the question.

    The video is one ``input_video`` part (``video_url``: any URL the model server accepts, e.g. a file or data URL)
    or labelled frames (``frames``: ``(label, image_url)`` pairs, see video_mcqa's ``video_parts``).
    """
    for name, value in (("task_id", task_id), ("question", question)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    return {
        "task_id": task_id,
        "setting": "single_turn",
        "responses_create_params": {
            "tools": [],
            "tool_choice": "none",
            "input": [
                {
                    "role": "user",
                    "content": [
                        *video_parts(video_url=video_url, frames=frames),
                        {"type": "input_text", "text": PROMPT.format(question=question)},
                    ],
                }
            ],
        },
        "verifier_metadata": {"question": question, "rubric": validate_rubric(rubric)},
    }
