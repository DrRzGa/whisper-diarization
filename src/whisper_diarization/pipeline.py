import argparse
import logging
import os
import re
import torch
import faster_whisper
from dataclasses import dataclass
from typing import Any

# Third-party logic imports
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from helpers import (
    cleanup,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)
from demucs.api import Separator
from tokenizers import Tokenizer
from diarization import MSDDDiarizer


@dataclass(frozen=True)
class PipelineConfig:
    """Constants and configuration strings for the pipeline."""
    PUNCT_MODEL: str = "kredor/punctuate-all"
    DEMUCS_MODEL: str = "htdemucs"
    DEFAULT_WHISPER_MODEL: str = "medium.en"
    TEMP_DIR_PREFIX: str = "temp_outputs"
    ENDING_PUNCTS: str = ".?!"
    MODEL_PUNCTS: str = ".,;:!?"
    ACRONYM_PATTERN: str = r"\b(?:[a-zA-Z]\.){2,}"


class TranscriptionEngine:
    """
    Modular engine to handle audio vocal isolation, transcription,
    forced alignment, and speaker diarization.
    """

    def __init__(
            self,
            model_name: str = PipelineConfig.DEFAULT_WHISPER_MODEL,
            device: str | None = None,
            batch_size: int = 8,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.batch_size = batch_size
        self.compute_type = self._determine_compute_type()

        # State placeholders
        self.audio_waveform: Any = None
        self.language_info: Any = None
        self.tokens_to_supress: Tokenizer | None = None
        self.suppress_numerals: bool = False
        self.transcribe_model = self.make_transcribe_model()
        self.diarizer = MSDDDiarizer(device=self.device)
        self.alignment_model, self.alignment_tokenizer = load_alignment_model(
            self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.punct_model = PunctuationModel(model=PipelineConfig.PUNCT_MODEL)


    def _determine_compute_type(self) -> str:
        """Checks GPU hardware capabilities to select the optimal compute type."""
        if self.device == "cpu":
            return "int8"

        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            # Compute Capability 7.0+ (Volta, Turing, Ampere, etc.) supports FP16 well.
            if major >= 7:
                logging.info(f"GPU detected with Compute Capability {major}. Using float16.")
                return "float16"

            logging.warning(f"Old GPU detected (Compute Capability {major}). Falling back to float32.")
        return "float32"

    def make_transcribe_model(self) -> faster_whisper.WhisperModel |  faster_whisper.BatchedInferencePipeline:
        model = faster_whisper.WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        self.tokens_to_supress =  find_numeral_symbol_tokens(model.hf_tokenizer) if self.suppress_numerals else [-1]
        if self.batch_size > 0:
            return faster_whisper.BatchedInferencePipeline(model)
        return model

    def prepare_audio(self, audio_path: str, stemming: bool, temp_dir: str) -> None:
        """Stage 1: Isolate vocals (optional) and load waveform into memory."""
        vocal_target = self._isolate_vocals(audio_path, temp_dir) if stemming else audio_path
        self.audio_waveform = faster_whisper.decode_audio(vocal_target)


    def transcribe(self, language: str | None, suppress_numerals: bool) -> str:
        """Stage 2: Run Whisper transcription and return the raw text."""
        # Correctly handle language processing
        lang_code = process_language_arg(language, self.model_name)

        if isinstance(self.transcribe_model, faster_whisper.BatchedInferencePipeline):
            segments, self.language_info = self.transcribe_model.transcribe(
                self.audio_waveform, lang_code, suppress_tokens=self.tokens_to_supress, batch_size=self.batch_size
            )
        else:
            segments, self.language_info = self.transcribe_model.transcribe(
                self.audio_waveform, lang_code, suppress_tokens=self.tokens_to_supress, vad_filter=True
            )

        full_transcript = "".join(s.text for s in segments)

        # del model, pipeline
        # torch.cuda.empty_cache()
        return full_transcript

    def align_and_diarize(self, transcript: str) -> tuple[list[dict], list]:
        """Stage 3: Align text to audio and identify speakers."""
        word_timestamps = self._run_alignment(transcript)
        speaker_ts = self.diarizer.diarize(torch.from_numpy(self.audio_waveform).unsqueeze(0))
        # del diarizer
        # torch.cuda.empty_cache()
        return word_timestamps, speaker_ts

    def post_process(self, word_timestamps: list[dict], speaker_ts: list) -> list[dict]:
        """Stage 4: Combine alignment with speaker data and restore punctuation."""
        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
        wsm = self._restore_punctuation(wsm)
        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        return get_sentences_speaker_mapping(wsm, speaker_ts)

    def run_full_pipeline(
            self, audio_path: str,
            language: str | None = None,
            stemming: bool = True,
            suppress_numerals: bool = False
    ) -> list[dict]:
        """Orchestrates the granular stages into a single workflow."""
        pid_dir = f"{PipelineConfig.TEMP_DIR_PREFIX}_{os.getpid()}"

        try:
            self.prepare_audio(audio_path, stemming, pid_dir)
            transcript = self.transcribe(language, suppress_numerals)
            word_ts, speaker_ts = self.align_and_diarize(transcript)
            return self.post_process(word_ts, speaker_ts)
        finally:
            # cleanup(os.path.join(os.getcwd(), PipelineConfig.TEMP_DIR_PREFIX))
            pass
    # --- Private Internal Helpers ---

    def _isolate_vocals(self, audio_path: str, output_dir: str) -> str:
        logging.info("Isolating vocals via Demucs...")
        cmd = (
            f'python -m demucs.separate -n {PipelineConfig.DEMUCS_MODEL} --two-stems=vocals '
            f'"{audio_path}" -o "{output_dir}" --device "{self.device}"'
        )
        if os.system(cmd) != 0:
            return audio_path

        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        vocal_path = os.path.join(output_dir, PipelineConfig.DEMUCS_MODEL, base_name, "vocals.wav")
        return vocal_path if os.path.exists(vocal_path) else audio_path



    def _run_alignment(self, transcript: str) -> list[dict]:


        emissions, stride = generate_emissions(
            self.alignment_model,
            torch.from_numpy(self.audio_waveform).to(self.alignment_model.dtype).to(self.device),
            batch_size=self.batch_size
        )

        # del model
        # torch.cuda.empty_cache()

        tokens_starred, text_starred = preprocess_text(
            transcript, romanize=True, language=langs_to_iso[self.language_info.language]
        )
        segments, scores, blank_token = get_alignments(emissions, tokens_starred, self.alignment_tokenizer)
        spans = get_spans(tokens_starred, segments, blank_token)
        return postprocess_results(text_starred, spans, stride, scores)

    def _restore_punctuation(self, wsm: list[dict]) -> list[dict]:
        if self.language_info.language not in punct_model_langs:
            return wsm

        words_list = [x["word"] for x in wsm]
        labeled_words = self.punct_model.predict(words_list, chunk_size=230)

        is_acronym = lambda x: re.fullmatch(PipelineConfig.ACRONYM_PATTERN, x)

        for word_dict, labeled_tuple in zip(wsm, labeled_words):
            word = word_dict["word"]
            if (word and labeled_tuple[1] in PipelineConfig.ENDING_PUNCTS and
                    (word[-1] not in PipelineConfig.MODEL_PUNCTS or is_acronym(word))):
                word_dict["word"] = f"{word}{labeled_tuple[1]}".replace("..", ".")

        return wsm


# --- CLI Entry Point ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--audio", required=True)
    parser.add_argument("--no-stem", action="store_false", dest="stemming", default=True)
    parser.add_argument("--suppress_numerals", action="store_true", default=False)
    parser.add_argument("--whisper-model", default="medium.en")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--language", type=str, default=None, choices=whisper_langs)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    engine = TranscriptionEngine(model_name=args.whisper_model, device=args.device, batch_size=args.batch_size)

    ssm = engine.run_full_pipeline(
        audio_path=args.audio,
        language=args.language,
        stemming=args.stemming,
        suppress_numerals=args.suppress_numerals
    )

    base = os.path.splitext(args.audio)[0]
    with open(f"{base}.txt", "w", encoding="utf-8-sig") as f: get_speaker_aware_transcript(ssm, f)
    with open(f"{base}.srt", "w", encoding="utf-8-sig") as srt: write_srt(ssm, srt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()