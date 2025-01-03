from pathlib import Path

from src.datasets.base_dataset import BaseDataset

from speechbrain.inference.TTS import Tacotron2


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, data_path, text_from_console=None, *args, **kwargs):
        self.data = []
        self.tacotron = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts", overrides={"max_decoder_steps": 10000})
        if text_from_console is None:
            for path in Path(data_path).iterdir():
                entry = {}
                if path.suffix in [".txt"]:
                    entry["path"] = str(path)
                    transc_path = Path(data_path) / (path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["text"] = f.read().strip()
                    else:
                        entry["text"] = "-"
                if len(entry) > 0:
                    self.data.append(entry)
        else:
            entry = {}
            entry['text'] = text_from_console
            entry['path'] = None
            self.data.append(entry)
        super().__init__(self.data, *args, **kwargs)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        path = self.data[idx]['path']
        generated_text_melspec, _, _ = self.tacotron.encode_text(text)
        return {'generated_text_melspec' : generated_text_melspec, 'path': path}