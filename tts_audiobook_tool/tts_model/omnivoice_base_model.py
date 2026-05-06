# tts_audiobook_tool/tts_model/omnivoice_base_model.py
from __future__ import annotations

from tts_audiobook_tool.tts_model.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos


class OmniVoiceBaseModel(TtsBaseModel):

    INFO = TtsModelInfos.OMNIVOICE.value

    DEFAULT_REPO_ID  = "k2-fsa/OmniVoice"
    SAMPLE_RATE      = 24_000
    DEFAULT_SPEED    = 1.0
    DEFAULT_NUM_STEP = 32

    def __init__(self) -> None:
        raise NotImplementedError

    def generate_using_project(self, project, prompts, force_random_seed=False, **kwargs):
        raise NotImplementedError

    def kill(self) -> None:
        raise NotImplementedError