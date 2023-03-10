import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import soundfile as sf



def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


hps = utils.get_hparams_from_file("./configs/biaobei_base.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint('G_1434000.pth', net_g, None)

text = "我是VITS本地推理版"  # @param {type: 'string'}
length_scale = 1  # @param {type:"slider", min:0.1, max:3, step:0.05}
filename = 'newtest'  # @param {type: "string"}
audio_path = f'{filename}.wav'
stn_tst = get_text(text, hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=length_scale)[0][
        0, 0].data.cpu().float().numpy()
# ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate))
sf.write(audio_path, audio, samplerate=hps.data.sampling_rate)
