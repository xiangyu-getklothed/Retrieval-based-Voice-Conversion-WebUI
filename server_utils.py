#   type: 
from multiprocessing import cpu_count
import threading
from time import sleep
from subprocess import Popen
from time import sleep
import torch, os, traceback, sys, warnings, shutil, numpy as np
import faiss
from random import shuffle

now_dir = os.path.dirname(__file__)
sys.path.append(now_dir)
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
import ffmpeg

# check gpu availability
ncpu = cpu_count()
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if (not torch.cuda.is_available()) or ngpu == 0:
    if_gpu_ok = False
else:
    if_gpu_ok = False
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if (
            "10" in gpu_name
            or "16" in gpu_name
            or "20" in gpu_name
            or "30" in gpu_name
            or "40" in gpu_name
            or "A2" in gpu_name.upper()
            or "A3" in gpu_name.upper()
            or "A4" in gpu_name.upper()
            or "P4" in gpu_name.upper()
            or "A50" in gpu_name.upper()
            or "70" in gpu_name
            or "80" in gpu_name
            or "90" in gpu_name
            or "M4" in gpu_name.upper()
            or "T4" in gpu_name.upper()
            or "TITAN" in gpu_name.upper()
        ):  # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # at least one Nvidia GPU
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok == True and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = "Sorry, no Nvidia GPU found."
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])
from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from scipy.io import wavfile
from fairseq import checkpoint_utils
import logging
from vc_infer_pipeline import VC
from config import Config
from infer_uvr5 import _audio_pre_
from my_utils import load_audio
from train.process_ckpt import show_info, change_info, merge, extract_small_model

config = Config()
# from trainset_preprocess_pipeline import PreProcess
logging.getLogger("numba").setLevel(logging.WARNING)

hubert_model = None


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [os.path.join(now_dir, "hubert_base.pt")],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


weight_root = os.path.join(now_dir, "weights")
weight_uvr5_root = os.path.join(now_dir, "uvr5_weights")
index_root = os.path.join(now_dir, "logs")
names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth"):
        uvr5_names.append(name.replace(".pth", ""))


def vc_single(
    sid,
    input_audio_path,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global tgt_sr, net_g, vc, hubert_model
    if input_audio_path is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        audio = load_audio(input_audio_path, 16000)
        times = [0, 0, 0]
        if hubert_model == None:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
            else file_index2
        )  # fix typos in file_index
        # file_big_npy = (
        #     file_big_npy.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        # )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            f0_file=f0_file,
        )
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)


def vc_multi(
    sid,
    dir_path,
    opt_root,
    paths,
    f0_up_key,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
):
    try:
        dir_path = (
            dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # fix typo in dir_path
        opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        os.makedirs(opt_root, exist_ok=True)
        try:
            if dir_path != "":
                paths = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
            else:
                paths = [path.name for path in paths]
        except:
            traceback.print_exc()
            paths = [path.name for path in paths]
        infos = []
        for path in paths:
            info, opt = vc_single(
                sid,
                path,
                f0_up_key,
                None,
                f0_method,
                file_index,
                file_index2,
                # file_big_npy,
                index_rate,
                filter_radius,
                resample_sr,
            )
            if "Success" in info:
                try:
                    tgt_sr, audio_opt = opt
                    wavfile.write(
                        "%s/%s" % (opt_root, os.path.basename(path)), tgt_sr, audio_opt
                    )
                except:
                    info += traceback.format_exc()
            infos.append("%s->%s" % (os.path.basename(path), info))
            yield "\n".join(infos)
        yield "\n".join(infos)
    except:
        yield traceback.format_exc()


def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg):
    infos = []
    try:
        inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        pre_fun = _audio_pre_(
            agg=int(agg),
            model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
            device=config.device,
            is_half=config.is_half,
        )
        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [path.name for path in paths]
        for path in paths:
            inp_path = os.path.join(inp_root, path)
            need_reformat = 1
            done = 0
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if (
                    info["streams"][0]["channels"] == 2
                    and info["streams"][0]["sample_rate"] == "44100"
                ):
                    need_reformat = 0
                    pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal)
                    done = 1
            except:
                need_reformat = 1
                traceback.print_exc()
            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (tmp, os.path.basename(inp_path))
                os.system(
                    "ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y"
                    % (inp_path, tmp_path)
                )
                inp_path = tmp_path
            try:
                if done == 0:
                    pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal)
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                yield "\n".join(infos)
            except:
                infos.append(
                    "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                )
                yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            del pre_fun.model
            del pre_fun
        except:
            traceback.print_exc()
        print("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    yield "\n".join(infos)


# Only one voice can be extracted at a time
def get_vc(sid):
    global n_spk, tgt_sr, net_g, vc, cpt
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model != None:  # check whether the model is available for each sid
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ### data cleaning
            if_f0 = cpt.get("f0", 1)
            if if_f0 == 1:
                net_g = SynthesizerTrnMs256NSFsid(
                    *cpt["config"], is_half=config.is_half
                )
            else:
                net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return {"visible": False, "__type__": "update"}
    person = "%s/%s" % (weight_root, sid)
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    if if_f0 == 1:
        net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
    else:
        net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))  # data cleaning
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return {"visible": True, "maximum": n_spk, "__type__": "update"}


def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }


def clean():
    return {"value": "", "__type__": "update"}


def change_f0(if_f0_3, sr2):  # np7, f0method8,pretrained_G14,pretrained_D15
    if if_f0_3:
        return (
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            "pretrained/f0G%s.pth" % sr2,
            "pretrained/f0D%s.pth" % sr2,
        )
    return (
        {"visible": False, "__type__": "update"},
        {"visible": False, "__type__": "update"},
        "pretrained/G%s.pth" % sr2,
        "pretrained/D%s.pth" % sr2,
    )


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def if_done(done, p):
    while 1:
        if p.poll() == None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:        
        # poll==None means the process is still running
        # it won't end until all processes are done
        flag = 1
        for p in ps:
            if p.poll() == None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    cmd = (
        config.python_cmd
        + " trainset_preprocess_pipeline_print.py %s %s %s %s/logs/%s "
        % (trainset_dir, sr, n_p, now_dir, exp_dir)
        + str(config.noparallel)
    )
    print(cmd)
    p = Popen(cmd, shell=True)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE,cwd=now_dir
    # When using gradio, all the processes have to finish running completely before reading all at once. 
    # Without gradio, it can read one line of output at a time normally. 
    # Only option is to create an additional text stream for periodic reading.
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0] == True:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    print(log)
    yield log


# but2.click(extract_f0,[gpus6,np7,f0method8,if_f0_3,trainset_dir4],[info2])
def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir):
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        cmd = config.python_cmd + " extract_f0_print.py %s/logs/%s %s %s" % (
            now_dir,
            exp_dir,
            n_p,
            f0method,
        )
        print(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
        # When using gradio, all the processes have to finish running completely before reading all at once. 
        # Without gradio, it can read one line of output at a time normally. 
        # Only option is to create an additional text stream for periodic reading.
        done = [False]
        threading.Thread(
            target=if_done,
            args=(
                done,
                p,
            ),
        ).start()
        while 1:
            with open(
                "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield (f.read())
            sleep(1)
            if done[0] == True:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        print(log)
        yield log
    #### use multi-processes for different parts
    """
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    """
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = config.python_cmd + " extract_feature_print.py %s %s %s %s %s/logs/%s" % (
            config.device,
            leng,
            idx,
            n_g,
            now_dir,
            exp_dir,
        )
        print(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    # When using gradio, all the processes have to finish running completely before reading all at once. 
    # Without gradio, it can read one line of output at a time normally. 
    # Only option is to create an additional text stream for periodic reading.
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0] == True:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    print(log)
    yield log


def change_sr2(sr2, if_f0_3):
    if if_f0_3:
        return "pretrained/f0G%s.pth" % sr2, "pretrained/f0D%s.pth" % sr2
    else:
        return "pretrained/G%s.pth" % sr2, "pretrained/D%s.pth" % sr2


# but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])
def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
):
    # generate filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    co256_dir = "%s/3_feature256" % (exp_dir)
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(co256_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(co256_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    co256_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    co256_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature256/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature256/mute.npy|%s"
                % (now_dir, sr2, now_dir, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    print("write filelist done")
    # generate config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    print("use gpus:", gpus16)
    if gpus16:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s -pg %s -pd %s -l %s -c %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                pretrained_G14,
                pretrained_D15,
                1 if if_save_latest13 == "yes" else 0,
                1 if if_cache_gpu17 == "yes" else 0,
            )
        )
    else:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s -pg %s -pd %s -l %s -c %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                pretrained_G14,
                pretrained_D15,
                1 if if_save_latest13 == "yes" else 0,
                1 if if_cache_gpu17 == "yes" else 0,
            )
        )
    print(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return "Training completes, you can check train.log under /logs"


# but4.click(train_index, [exp_dir1], info3)
def train_index(exp_dir1):
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = "%s/3_feature256" % (exp_dir)
    if os.path.exists(feature_dir) == False:
        return "Please extract features first!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "Please extract features first!"
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    # n_ivf =  big_npy.shape[0] // 39
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos = []
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s.index" % (exp_dir, n_ivf, index_ivf.nprobe),
    )
    # faiss.write_index(index, '%s/trained_IVF%s_Flat_FastScan.index'%(exp_dir,n_ivf))
    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s.index" % (exp_dir, n_ivf, index_ivf.nprobe),
    )
    infos.append("Succesfully created the index，added_IVF%s_Flat_nprobe_%s.index" % (n_ivf, index_ivf.nprobe))
    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan.index'%(exp_dir,n_ivf))
    # infos.append("Succesfully created the index，added_IVF%s_Flat_FastScan.index"%(n_ivf))
    yield "\n".join(infos)


# but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key(
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir4,
    spk_id5,
    np7,
    f0method8,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
):
    infos = []

    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)

    model_log_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    preprocess_log_path = "%s/preprocess.log" % model_log_dir
    extract_f0_feature_log_path = "%s/extract_f0_feature.log" % model_log_dir
    gt_wavs_dir = "%s/0_gt_wavs" % model_log_dir
    feature256_dir = "%s/3_feature256" % model_log_dir

    os.makedirs(model_log_dir, exist_ok=True)
    #########step1: Processing data
    open(preprocess_log_path, "w").close()
    cmd = (
        config.python_cmd
        + " %s/trainset_preprocess_pipeline_print.py %s %s %s %s "
        % (now_dir, trainset_dir4, sr_dict[sr2], np7, model_log_dir)
        + str(config.noparallel)
    )
    yield get_info_str("step1: Processing data")
    yield get_info_str(cmd)
    p = Popen(cmd, shell=True)
    p.wait()
    with open(preprocess_log_path, "r") as f:
        print(f.read())
    #########step2a: Extract pitch
    open(extract_f0_feature_log_path, "w")
    if if_f0_3:
        yield get_info_str("step2a: Extracting pitch")
        cmd = config.python_cmd + " %s/extract_f0_print.py %s %s %s" % (
            now_dir,
            model_log_dir,
            np7,
            f0method8,
        )
        yield get_info_str(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)
        p.wait()
        with open(extract_f0_feature_log_path, "r") as f:
            print(f.read())
    else:
        yield get_info_str("step2a: No need to extract pitch")
    #######step2b: Extract features
    yield get_info_str("step2b: Extracting features")
    gpus = gpus16.split("-")
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = config.python_cmd + " %s/extract_feature_print.py %s %s %s %s %s" % (
            now_dir,
            config.device,
            leng,
            idx,
            n_g,
            model_log_dir,
        )
        yield get_info_str(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    for p in ps:
        p.wait()
    with open(extract_f0_feature_log_path, "r") as f:
        print(f.read())
    #######step3a: Train the model
    yield get_info_str("step3a: Training the model")
    # Generate filelist
    if if_f0_3:
        f0_dir = "%s/2a_f0" % model_log_dir
        f0nsf_dir = "%s/2b-f0nsf" % model_log_dir
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature256_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature256_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature256_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature256_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature256/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature256/mute.npy|%s"
                % (now_dir, sr2, now_dir, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % model_log_dir, "w") as f:
        f.write("\n".join(opt))
    yield get_info_str("write filelist done")
    if gpus16:
        cmd = (
            config.python_cmd
            + " %s/train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s -pg %s -pd %s -l %s -c %s"
            % (
                now_dir,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                pretrained_G14,
                pretrained_D15,
                1 if if_save_latest13 == "yes" else 0,
                1 if if_cache_gpu17 == "yes" else 0,
            )
        )
    else:
        cmd = (
            config.python_cmd
            + " %s/train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s -pg %s -pd %s -l %s -c %s"
            % (
                now_dir,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                pretrained_G14,
                pretrained_D15,
                1 if if_save_latest13 == "yes" else 0,
                1 if if_cache_gpu17 == "yes" else 0,
            )
        )
    yield get_info_str(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    yield get_info_str("yes")
    #######step3b: Train the index
    npys = []
    listdir_res = list(os.listdir(feature256_dir))
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature256_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)

    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    np.save("%s/total_fea.npy" % model_log_dir, big_npy)

    # n_ivf =  big_npy.shape[0] // 39
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    yield get_info_str("%s,%s" % (big_npy.shape, n_ivf))
    index = faiss.index_factory(256, "IVF%s,Flat" % n_ivf)
    yield get_info_str("training index")
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s.index"
        % (model_log_dir, n_ivf, index_ivf.nprobe),
    )
    yield get_info_str("adding index")
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s.index"
        % (model_log_dir, n_ivf, index_ivf.nprobe),
    )
    yield get_info_str(
        "Successfully trained the index, added_IVF%s_Flat_nprobe_%s.index" % (n_ivf, index_ivf.nprobe)
    )
    yield get_info_str("The whole pipeline completes！")


#   ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__])
def change_info_(ckpt_path):
    if (
        os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log"))
        == False
    ):
        return {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(
            ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r"
        ) as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            return sr, str(f0)
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}


from infer_pack.models_onnx_moess import SynthesizerTrnMs256NSFsidM
from infer_pack.models_onnx import SynthesizerTrnMs256NSFsidO


def export_onnx(ModelPath, ExportedPath, MoeVS=True):
    hidden_channels = 256  # hidden_channels，prepare for 768Vec
    cpt = torch.load(ModelPath, map_location="cpu")
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    print(*cpt["config"])

    test_phone = torch.rand(1, 200, hidden_channels)  # hidden unit
    test_phone_lengths = torch.tensor([200]).long()  # hidden unit length (doesn't make any sense)
    test_pitch = torch.randint(size=(1, 200), low=5, high=255)  # base frequency(Hz)
    test_pitchf = torch.rand(1, 200)  # nsf base frequency
    test_ds = torch.LongTensor([0])  # speaker id
    test_rnd = torch.rand(1, 192, 200)  # noise for randomnization

    device = "cpu"  # device for export

    if MoeVS:
        net_g = SynthesizerTrnMs256NSFsidM(
            *cpt["config"], is_half=False
        )  # FP32 export (C++ requires manual memory rearrangement to support FP16, so FP16 is temporarily not used).
        net_g.load_state_dict(cpt["weight"], strict=False)
        input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
        output_names = [
            "audio",
        ]
        torch.onnx.export(
            net_g,
            (
                test_phone.to(device),
                test_phone_lengths.to(device),
                test_pitch.to(device),
                test_pitchf.to(device),
                test_ds.to(device),
                test_rnd.to(device),
            ),
            ExportedPath,
            dynamic_axes={
                "phone": [1],
                "pitch": [1],
                "pitchf": [1],
                "rnd": [2],
            },
            do_constant_folding=False,
            opset_version=16,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
        )
    else:
        net_g = SynthesizerTrnMs256NSFsidO(
            *cpt["config"], is_half=False
        )  # FP32 export (C++ requires manual memory rearrangement to support FP16, so FP16 is temporarily not used).
        net_g.load_state_dict(cpt["weight"], strict=False)
        input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds"]
        output_names = [
            "audio",
        ]
        torch.onnx.export(
            net_g,
            (
                test_phone.to(device),
                test_phone_lengths.to(device),
                test_pitch.to(device),
                test_pitchf.to(device),
                test_ds.to(device),
            ),
            ExportedPath,
            dynamic_axes={
                "phone": [1],
                "pitch": [1],
                "pitchf": [1],
            },
            do_constant_folding=False,
            opset_version=16,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
        )
    return "Finished"

def model_inference_single(model_path, index_path, audio_path, save_path, error_log_path, pitch_shift=0):
    # import refrence timbre model
    sid0 = clean() # clean the cache
    sid0, file_index = change_choices() # refresh the choices
    spk_item = 0 # speaker id
    assert os.path.exists(audio_path), "audio file not found"
    assert model_path in sid0['choices'], "model not found"
    assert index_path in file_index['choices'], "index file not found"
    get_vc(model_path) # load the model
    print("%d speakers detected" % n_spk)
    
    f0method = "pm" # pitch extraction method, pm or harvest
    filter_radius = 3 # filter radius for pitch extraction
    index_rate = 0.76
    resample_sr = 0 # resample to this sample rate, 0 for no resample
    f0_file = None # f0 file path, optional

    info, opt = vc_single(spk_item,
            audio_path,
            pitch_shift,
            f0_file,
            f0method,
            "",
            index_path,
            index_rate,
            filter_radius,
            resample_sr,
            )

    if "Success" in info:
        try:
            tgt_sr, audio_opt = opt
            wavfile.write(
                save_path, tgt_sr, audio_opt
            )
        except:
            info += traceback.format_exc()
            with open(error_log_path, "w") as f:
                f.write(info)
    

def model_inference_multi(model_path, index_path, input_dir, log_path, pitch_shift=0):
    # import refrence timbre model
    opt_input = "opt" # folder path for opt output
    sid0 = clean() # clean the cache
    sid0, file_index = change_choices() # refresh the choices
    spk_item = 0 # speaker id
    if model_path not in sid0['choices']:
        print("model not found, please check the model path")
        return
    if index_path not in file_index['choices']:
        print("index file not found, please check the index path")
        return
    get_vc(model_path) # load the model
    print("%d speakers detected" % n_spk)

    f0method = "pm" # pitch extraction method, pm or harvest
    filter_radius = 3 # filter radius for pitch extraction
    index_rate = 0.76
    resample_sr = 0 # resample to this sample rate, 0 for no resample
    
    vc_outputs = vc_multi(
        spk_item,
        input_dir,
        opt_input,
        [],
        pitch_shift,
        f0method,
        "",
        file_index,
        index_rate,
        filter_radius,
        resample_sr,
    )

    infos = []
    for vc_output in vc_outputs:
        infos.append(vc_output)
        with open(log_path, "w") as f:
            f.write("\n".join(infos))
                
def vocal_separation(dir_wav_input, log_path):
    model_choose = uvr5_names[0] # HP5 or HP2, two models in total
    agg = 10 # aggressiveness of vocal separation
    opt_vocal_root = "opt" # folder path for vocal results
    opt_ins_root = "opt" # folder path for instrumental results
    vc_outputs = uvr(model_choose,
                dir_wav_input,
                opt_vocal_root,
                "",
                opt_ins_root,
                agg,)
    infos = []
    for vc_output in vc_outputs:
        infos.append(vc_output)
        with open(log_path, "w") as f:
            f.write("\n".join(infos))
    

def merge_model(model_A_path, model_B_path, alpha, save_path="output", log_path = "log.txt"):
    sr = "40k" # choices=["32k", "40k", "48k"]
    if_f0 = True # whether the model has pitch guidance
    save_path = "output" # folder path for saving the merged model
    info = merge(model_A_path, model_B_path, alpha, sr, if_f0, "", save_path)
    with open(log_path, "w") as f:
        f.write(info)

def extract_model(model_path, save_path, log_path):
    sr = "40k" # choices=["32k", "40k", "48k"]
    if_f0 = True # whether the model has pitch guidance
    info = extract_small_model(model_path, save_path, sr, if_f0, "")
    with open(log_path, "w") as f:
        f.write(info)

def train_model(exp_name, trainset_dir, log_path, total_epoch=100):
    sr = "40k" # choices=["32k", "40k", "48k"]
    if_f0 = True # whether the model has pitch guidance
    np = 8 # number of processes for pitch extraction and data processing
    spk_id = 0 # speaker id

    f0method = "harvest" # pitch extraction method, pm or harvest or dio
    save_epoch = 10 # save model every x epochs
    batch_size = default_batch_size
    if_save_latest = "yes" # whether to save the latest model, save space
    if_cache_gpu = "yes" # whether to cache the gpu data, improve the speed for <10min data

    pretrained_G = "pretrained/f0G40k.pth" # pretrained generator path
    pretrained_D = "pretrained/f0D40k.pth" # pretrained discriminator path
    if sr != "40k":
        pretrained_G, pretrained_D = change_sr2(sr, if_f0)
    if if_f0 == False:
        np, f0method, pretrained_G, pretrained_D = change_f0(if_f0, sr)
    infos = train1key(exp_name,
            sr,
            if_f0,
            trainset_dir,
            spk_id,
            np,
            f0method,
            save_epoch,
            total_epoch,
            batch_size,
            if_save_latest,
            pretrained_G,
            pretrained_D,
            gpus,
            if_cache_gpu,)
    with open(log_path, "w") as f:
        for info in infos:
            f.write(info + '\n')
    

if __name__ == "__main__":
    exp_name = "drake-100"
    trainset_dir = "/home/fantasyfish/Desktop/dotdemo/examples/drake"
    log_path = "train_log.txt"
    train_model(exp_name, trainset_dir, log_path)

    model_path = "drake-200.pth"
    index_path = "logs/drake-200/added_IVF619_Flat_nprobe_1.index"
    audio_path = "/home/fantasyfish/Desktop/dotdemo/examples/zefaan/bliss ai zz.mp3"
    save_path = "bliss ai zz_drake-200.wav"
    error_log_path = "error_log.txt"
    pitch_shift=0
    # model_inference_single(model_path, index_path, audio_path, save_path, error_log_path, pitch_shift)