# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The agentic video harness configs resolve as Gym loads them, at both budgets, with the settings they fix."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from resources_servers.video_mcqa.app import VideoMCQAResourcesServerConfig
from resources_servers.video_mcqa.sandbox import PLUGIN_PATH
from resources_servers.video_qa_rubric.app import VideoQARubricResourcesServerConfig
from resources_servers.videomme_v2.app import VideoMMEV2ResourcesServerConfig
from responses_api_agents.codex_agent.app import CodexAgentConfig
from responses_api_agents.opencode_agent.app import OpenCodeAgentConfig
from responses_api_agents.pi_agent.app import PiAgentConfig
from responses_api_models.vllm_model.app import VLLMModelConfig


RESOURCES = Path(__file__).parents[2]
HARNESS_IMAGE = RESOURCES / "video_mcqa" / "sandbox_image_harnesses"
BUDGETS = {"256k": (262_144, 32_768), "512k": (524_288, 65_536)}
# Where each agent compacts at each budget: OpenCode with the video tools and with read alone (the context window
# it is told), Codex (its compaction limit).
COMPACTION = {"256k": (151_072, 232_992, 188_416), "512k": (380_448, 462_368, 385_024)}
RESOURCE_TYPES = {
    "videomme_v2": VideoMMEV2ResourcesServerConfig,
    "videomme_v1": VideoMCQAResourcesServerConfig,
    "lvbench": VideoMCQAResourcesServerConfig,
    "video_qa_rubric": VideoQARubricResourcesServerConfig,
}


def resolved(tmp_path, server: str, budget: str) -> dict:
    values = {
        "config_paths": [str(RESOURCES / server / "configs" / f"{server}_agentic_harnesses.yaml")],
        f"{server}_agentic_video_roots": [str(tmp_path)],
        f"{server}_agentic_cohort_manifest": str(tmp_path / "cohort.json"),
        "judge_base_url": "https://judge.example.invalid/v1",
        "judge_api_key": "placeholder",
        "judge_model_name": "judge-model",
        "video_agentic_budget": budget,
        "policy_model": {
            "responses_api_models": {
                "vllm_model": {
                    "entrypoint": "app.py",
                    "base_url": "http://policy.example.invalid/v1",
                    "api_key": "placeholder",
                    "model": "policy",
                    "return_token_id_information": False,
                    "uses_reasoning_parser": True,
                }
            }
        },
    }
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT, {**values, "policy_model_name": "policy"}
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    return OmegaConf.to_container(config, resolve=True)


@pytest.mark.parametrize("budget", sorted(BUDGETS))
@pytest.mark.parametrize("server", sorted(RESOURCE_TYPES))
def test_harness_configs_resolve_with_their_settings(tmp_path, monkeypatch, server, budget) -> None:
    monkeypatch.setenv("VIDEO_MCQA_SANDBOX_IMAGE", "registry.example/video-mcqa-sandbox@sha256:" + "0" * 64)
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "placeholder")
    config = resolved(tmp_path, server, budget)
    context, output = BUDGETS[budget]
    name = f"{server}_agentic"
    (resource_block,) = config[name]["resources_servers"].values()
    resource = RESOURCE_TYPES[server](name=name, **resource_block)
    # The lent sandbox reaches only the model server, and outlives the episode deadline below.
    assert resource.sandbox_network_access == "model_only"
    assert resource.sandbox_config.ttl_s == 12600
    # Gym's agent composition reads the declaration from the config, not from the server's config class.
    assert resource_block["allowed_agents"] == ["opencode_agent", "pi_agent", "codex_agent"]

    agents = {}
    for mode, config_type in (
        ("oc_video", OpenCodeAgentConfig),
        ("oc_read", OpenCodeAgentConfig),
        ("pi", PiAgentConfig),
        ("codex", CodexAgentConfig),
    ):
        (block,) = config[f"{name}_{mode}"]["responses_api_agents"].values()
        agents[mode] = config_type(name=f"{name}_{mode}", **block)
        environment = SingleAgentTurnLegacyEnvironmentServerConfig(
            name=f"{name}_{mode}_environment",
            **config[f"{name}_{mode}_environment"]["environment_servers"]["single_agent_turn_legacy"],
        )
        assert environment.agent_server.name == f"{name}_{mode}" and environment.resources_server.name == name
        # Two hours per run; the episode deadline leaves room for the sandbox start, the upload and grading.
        assert agents[mode].timeout == 7200 < environment.default_episode_timeout_seconds == 10800

    oc_video, oc_read, pi, codex = agents.values()
    # Both OpenCode setups are the benchmark's own OpenCode agent with compaction, the budget and the run time
    # changed; the read setup also leaves out the video tools plugin.
    (base,) = config[f"{name}_opencode"]["responses_api_agents"].values()
    (video_block,) = config[f"{name}_oc_video"]["responses_api_agents"].values()
    changed = {"description", "timeout", "context_window", "max_output_tokens", "opencode_config"}
    changed |= {"exclude_compaction_summaries", "datasets"}
    assert {k: v for k, v in video_block.items() if k not in changed} == {
        k: v for k, v in base.items() if k not in changed
    }
    assert oc_video.plugins == [str(PLUGIN_PATH)] and oc_read.plugins == []
    # The engine serves the budget's context and refuses a call whose prompt and output cap together exceed it.
    # OpenCode compacts after a call whose tokens reach its context window minus 20,000, Codex at its limit; both
    # must do so while the next step still fits: one watch_video result adds up to about 96,000 tokens, a read step
    # about 13,000, and Codex's compaction call carries the whole history with another output cap.
    refused_above = context - output
    oc_video_window, oc_read_window, codex_limit = COMPACTION[budget]
    assert (oc_video.context_window, oc_read.context_window) == (oc_video_window, oc_read_window)
    assert oc_video.context_window - 20_000 == refused_above - 98_304
    assert oc_read.context_window - 20_000 == refused_above - 16_384
    assert codex.model_auto_compact_token_limit == codex_limit == context - 2 * output - 8_192
    for agent in (oc_video, oc_read):
        assert agent.max_output_tokens == output
        assert agent.opencode_config["compaction"] == {"auto": True} and agent.exclude_compaction_summaries
        assert agent.thinking and agent.image_input
        assert agent.opencode_config["tools"] == {"webfetch": False}
        assert agent.opencode_config["permission"]["doom_loop"] == "ask"
        assert agent.opencode_config == {**base["opencode_config"], "compaction": {"auto": True}}

    assert (pi.context_window, pi.max_output_tokens) == (context, output)
    assert pi.auto_compaction and pi.thinking == "high" and pi.image_input
    assert codex.model_context_window == context
    assert codex.reasoning_effort == "high" and codex.model_server_forwards_reasoning_effort
    assert codex.features == {"multi_agent": False, "code_mode": False}

    # The policy model server passes the engine's context-length refusal on, so that a harness can compact.
    policy = config["policy_model"]["responses_api_models"]["vllm_model"]
    assert VLLMModelConfig(name="policy_model", **policy).propagate_context_overflow_errors
    # Codex's model server is the policy's with Codex's conversions and the per-call output cap.
    codex_model_block = config[codex.model_server.name]["responses_api_models"]["vllm_model"]
    codex_model = VLLMModelConfig(name=codex.model_server.name, **codex_model_block)
    assert codex_model.inline_tool_output_images and codex_model.drop_unrepresentable_request_fields
    assert codex_model.sampling_overrides == {"max_tokens": output}
    assert codex_model.propagate_context_overflow_errors
    assert {k: v for k, v in codex_model_block.items() if k in policy} == policy

    # The runtimes the configs name are the ones the harness image installs, at the pinned versions.
    dockerfile = (HARNESS_IMAGE / "Dockerfile").read_text()
    assert "R=/opt/harness-runtimes" in dockerfile
    assert pi.sandbox_runtime_dir == "/opt/harness-runtimes/pi"
    assert codex.sandbox_runtime_dir == "/opt/harness-runtimes/codex"
    pi_package = json.loads((HARNESS_IMAGE / "pi" / "package.json").read_text())
    codex_package = json.loads((HARNESS_IMAGE / "codex" / "package.json").read_text())
    assert pi_package["dependencies"] == {"@earendil-works/pi-coding-agent": pi.pi_version}
    assert codex_package["dependencies"] == {"@openai/codex": codex.codex_version}
