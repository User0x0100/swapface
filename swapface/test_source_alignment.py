"""CPU regression checks for mixed source alignment: python -m swapface.test_source_alignment."""
import ast
import copy
import tempfile
import tomllib
from pathlib import Path
from collections.abc import Sequence
from typing import Any
import os

import numpy as np
import torch
from PIL import Image

from .config import DEFAULT_DATA_CONFIG, resolve_train_config
from .dataloader import TrainingDataLoader
from .dataloader_common import ImageSource, LocalImagePool, build_image_pools, print_image_pools
from .experiment import config_sha256, create_run
from .train import _load_run_config
from misc.face_alignment import make_ffhq_to_arcface_112_grid, ffhq_to_arcface_112
from .dataloader_native import _RandomImagePairDataset

def main():
    raw=tomllib.loads((Path(__file__).parents[1]/'experiments/train.toml').read_text())
    legacy=resolve_train_config(raw)
    explicit=copy.deepcopy(raw)
    for section in ('src','dst'):
        for entry in explicit['data'][section]:
            entry['alignment']='ffhq'
    assert resolve_train_config(explicit)==legacy
    assert config_sha256(resolve_train_config(explicit))==config_sha256(legacy)
    mixed=copy.deepcopy(raw)
    mixed['data']['src'].append({'path':'asian','adjustment':-0.5,'alignment':'arcface'})
    assert resolve_train_config(mixed)['data']['src'][-1]['alignment']=='arcface'
    for bad in ('unknown',None,112,[]):
        invalid=copy.deepcopy(raw)
        invalid['data']['src'][0]['alignment']=bad
        try:resolve_train_config(invalid)
        except ValueError:pass
        else:raise AssertionError(f'accepted invalid alignment {bad}')
    invalid=copy.deepcopy(raw)
    invalid['data']['dst'][0]['alignment']='arcface'
    try:resolve_train_config(invalid)
    except ValueError:pass
    else:raise AssertionError('accepted ArcFace dst')

    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        source_config=root/'legacy.toml'
        source_config.write_text((Path(__file__).parents[1]/'experiments/train.toml').read_text())
        run=create_run(root/'runs',source_config,legacy,name='legacy')
        assert _load_run_config(run)==legacy
        assert config_sha256(_load_run_config(run))==config_sha256(legacy)
        ffhq=root/'ffhq';arc=root/'arc';dst=root/'dst'
        for p in (ffhq,arc,dst):p.mkdir()
        y,x=np.mgrid[:112,:112]
        pixels=np.stack((x*2,y*2,(x+y)%256),axis=-1).astype(np.uint8)
        for p in (ffhq,arc,dst):Image.fromarray(pixels).save(p/'ramp.png')
        cfg={**DEFAULT_DATA_CONFIG['loader'],**DEFAULT_DATA_CONFIG['augmentation'],**DEFAULT_DATA_CONFIG['sampling']}
        cfg.update(py_num_workers=0,brightness=0.0,contrast=0.0,saturation=0.0,flip_prob=0.0,same_prob=0.0,rotation_range=(0.0,0.0),scale_factor_range=(1.0,1.0),tx_range=(0.0,0.0),ty_range=(0.0,0.0))
        # 真实训练尺寸与 ArcFace 112 不同；高频像素暴露 112->256->112 的重采样。
        pixels[...,2]=((x+y)%2*255).astype(np.uint8)
        for p in (ffhq,arc,dst):Image.fromarray(pixels).save(p/'ramp.png')
        normalized=torch.from_numpy(pixels.copy()).permute(2,0,1).float().div(127.5).sub(1)
        for resolution in (256,512):
            torch.manual_seed(42)
            loader=TrainingDataLoader(batch_size=16,device=torch.device('cpu'),img_resolution=resolution,src=[ffhq,(arc,0.0,'arcface')],dst=[dst],**cfg)
            src,target,canonical,_theta,same,identity=loader.next()
            assert src.shape==target.shape==canonical.shape==(16,3,resolution,resolution)
            assert identity.shape==(16,3,112,112) and identity.dtype==torch.float32
            # 两个池使用同一张图；展示图相同，但身份裁剪应分成两组。
            old=ffhq_to_arcface_112(src,make_ffhq_to_arcface_112_grid(resolution,16,torch.device('cpu')))
            direct=normalized.unsqueeze(0).expand(16,-1,-1,-1)
            direct_match=(identity==direct).flatten(1).all(1)
            old_match=(identity==old).flatten(1).all(1)
            assert direct_match.any() and old_match.any()
            assert (direct_match|old_match).all() and not (direct_match&old_match).any()
            assert not same.any()
            # 每个分支都必须与其预期输入逐像素相同。
            torch.testing.assert_close(identity[direct_match],direct[direct_match],atol=0,rtol=0)
            torch.testing.assert_close(identity[old_match],old[old_match],atol=0,rtol=0)
        # same pair 取 FFHQ dst，忽略独立 ArcFace src 池。
        cfg['same_prob']=1.0;cfg['flip_prob']=0.0
        same_loader=TrainingDataLoader(batch_size=2,device=torch.device('cpu'),img_resolution=256,src=[(arc,0.0,'arcface')],dst=[dst],**cfg)
        same_src,_,same_canonical,_,same_mask,identity=same_loader.next()
        assert same_mask.all()
        torch.testing.assert_close(same_src,same_canonical)
        expected_same=ffhq_to_arcface_112(same_src,make_ffhq_to_arcface_112_grid(256,2,torch.device('cpu')))
        torch.testing.assert_close(identity,expected_same,atol=0,rtol=0)
        # 展示和身份分支共享同一次 src flip，原始 112 身份像素不发生插值。
        cfg['same_prob']=0.0;cfg['flip_prob']=1.0
        flip_loader=TrainingDataLoader(batch_size=2,device=torch.device('cpu'),img_resolution=256,src=[(arc,0.0,'arcface')],dst=[dst],**cfg)
        flipped,_,_,_,_,identity=flip_loader.next()
        expected=normalized.flip(-1).unsqueeze(0).expand(2,-1,-1,-1)
        torch.testing.assert_close(identity,expected,atol=0,rtol=0)
        display=_RandomImagePairDataset([(arc,0.0,'arcface')],[dst],256,0.0)._decode_resize(str(arc/'ramp.png'))
        expected_display=display.float().div(127.5).sub(1).flip(-1).unsqueeze(0).expand(2,-1,-1,-1)
        torch.testing.assert_close(flipped,expected_display,atol=0,rtol=0)
        # FFHQ 翻转仍发生在旧裁剪之前。
        flip_ffhq=TrainingDataLoader(batch_size=2,device=torch.device('cpu'),img_resolution=256,src=[ffhq],dst=[dst],**cfg)
        ffhq_src,_,_,_,_,identity=flip_ffhq.next()
        old=ffhq_to_arcface_112(ffhq_src,make_ffhq_to_arcface_112_grid(256,2,torch.device('cpu')))
        torch.testing.assert_close(identity,old,atol=0,rtol=0)
        # Exercise DALI Python source without importing the unavailable CUDA-only SDK.
        source_tree=ast.parse(Path(__file__).with_name('dataloader_dali.py').read_text())
        source_node=next(n for n in source_tree.body if isinstance(n,ast.ClassDef) and n.name=='_RandomImagePairSource')
        namespace: dict[str, Any] = {'np':np,'ndarray':np.ndarray,'os':os,'Sequence':Sequence,'Any':Any,'ImageSource':ImageSource,'LocalImagePool':LocalImagePool,'build_image_pools':build_image_pools,'print_image_pools':print_image_pools}
        exec(compile(ast.Module(body=[source_node],type_ignores=[]),'<DALI source>','exec'),namespace)
        source_type=namespace['_RandomImagePairSource']
        dali=source_type([(arc,0.0,'arcface')],[dst],0.0)
        encoded,_,same,arcface=dali(None)
        assert not bool(same) and bool(arcface)
        assert encoded.tobytes()==(arc/'ramp.png').read_bytes()
        dali_same=source_type([(arc,0.0,'arcface')],[dst],1.0)
        encoded,dst_encoded,same,arcface=dali_same(None)
        assert bool(same) and not bool(arcface)
        assert np.array_equal(encoded,dst_encoded)
    print('PASS: FFHQ defaults and legacy config hash; 112 source pixels at 256/512, unchanged FFHQ crop, coupled RGB/flip, same pairs and DALI source metadata')

if __name__=='__main__':
    main()
