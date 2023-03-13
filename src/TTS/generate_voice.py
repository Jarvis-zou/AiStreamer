from models.synthesizer.inference import Synthesizer
from models.encoder import inference as encoder
from models.vocoder.hifigan import inference as gan_vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play


def get_voice_clone(input_voice_path, input_text):
    """基于input_voice的声纹对input_text进行模拟，输出为使用克隆声音朗读input_text的音频wav"""
    encoder_wav = synthesizer.load_preprocess_wav(input_voice_path)  # preprocess wav file
    embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)  # generate voice embed
    embeds = [embed] * len(input_text)
    specs = synthesizer.synthesize_spectrograms(input_text, embeds)
    breaks = [spec.shape[1] for spec in specs]
    spec = np.concatenate(specs, axis=1)

    # generate voice
    generated_wav, output_sample_rate = vocoder.infer_waveform(spec)

    # adding breaks into voice
    b_ends = np.cumsum(np.array(breaks) * synthesizer.hparams.hop_size)
    b_starts = np.concatenate(([0], b_ends[:-1]))
    wavs = [generated_wav[start:end] for start, end in zip(b_starts, b_ends)]
    breaks = [np.zeros(int(0.15 * synthesizer.sample_rate))] * len(breaks)
    generated_wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

    # erase blank in voice
    generated_wav = encoder.preprocess_wav(generated_wav)

    # adjustment
    generated_wav = generated_wav / np.abs(generated_wav).max() * 0.97
    file_name = 'output.wav'
    sf.write(file_name, generated_wav, synthesizer.sample_rate)

    return generated_wav, synthesizer.sample_rate


if __name__ == '__main__':
    # load models
    encoder_path = Path("data/ckpt/encoder/pretrained.pt")
    synthesizer_path = Path("data/ckpt/synthesizer/pretrained-11-7-21_75k.pt")
    vocoder_path = Path("data/ckpt/vocoder/g_hifigan.pt")
    encoder.load_model(encoder_path)
    synthesizer = Synthesizer(synthesizer_path)
    vocoder = gan_vocoder
    vocoder.load_model(vocoder_path)

    # start synthesize voice
    spk_wav_path = r'C:\Users\ZouJiawei\Desktop\Advanced_explore\MockingBird\audio_data\train\wav\fengge\fengge001.wav'  # target voice sample to clone
    text = ['大家好', '我是二次元峰哥']
    generated_audio, sample_rate = get_voice_clone(spk_wav_path, text)

    # # play audio
    song = AudioSegment.from_wav('output.wav')
    play(song)