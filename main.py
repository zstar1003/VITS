import torch
import utils
from models import SynthesizerTrn
from text.symbols import symbols
import soundfile as sf


if __name__ == '__main__':
    text = "我是VITS本地推理版"
    length_scale = 1
    filename = 'results/result'
    audio_path = f'{filename}.mp3'
    # 创建模型，加载参数
    hps = utils.get_hparams_from_file("./configs/biaobei_base.json")
    model = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    model.eval()
    utils.load_checkpoint('model/Paimon.pth', model)

    stn_tst = utils.get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = model.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=length_scale)[0][
            0, 0].data.cpu().float().numpy()
    sf.write(audio_path, audio, samplerate=hps.data.sampling_rate)
