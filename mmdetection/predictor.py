import glob
import json
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any
from matplotlib import pyplot as plt
from mmdet.apis import DetInferencer
import imageio
import numpy as np
import tqdm


class MMPredictor():
    
    def __init__(self, config, weight=None, device='cuda:0') -> None:
        # Initialize the DetInferencer
        self.inferencer = DetInferencer(config, weights=weight, device=device)

        # Perform inference
        # inferencer('demo/demo.jpg', show=True)

    
    def __call__(self, imgs, bs=1) -> Any:
        if isinstance(imgs, list):
            bs = len(imgs)
            
        res = self.inferencer(imgs, pred_score_thr=0, batch_size=bs)
        
        """
        {
            'predictions' : [
                # Each instance corresponds to an input image
                {
                'labels': [...],  # int list of length (N, )
                'scores': [...],  # float list of length (N, )
                'bboxes': [...],  # 2d list of shape (N, 4), format: [min_x, min_y, max_x, max_y]
                },
                ...
            ],
            'visualization' : [
                array(..., dtype=uint8),
            ]
        }
        """
        predictions = res['predictions']

        return predictions


from slide_detect_tools.slide_grid_patch import SdpcReader
from multiprocessing.queues import Queue
import multiprocessing as mp, os
from multiprocessing import get_context

os.environ["LD_PRELOAD"]="/usr/lib/x86_64-linux-gnu/libffi.so.7"

class WSIPredictor():
    
    
    def __init__(self, configs, wieghts, devices=[0, 1], batch_size=1, crop_workers=16) -> None:
        self.devices = devices
        self.batch_size = batch_size
        self.wieghts = wieghts
        self.configs = configs
        self.region_queue = Queue(maxsize=crop_workers * 16, ctx=get_context())
        self.image_queue = Queue(maxsize=32, ctx=get_context())
        self.crop_workers = crop_workers
        self.pred_queue = Queue(maxsize=64, ctx=get_context())
        np.random.seed(0)
        
    
    def start_pred_process(self, configs, wieghts, devices=[0, 1], batch_size=1):
        inferencer_params = []
        for d in devices:
            inferencer = (configs, wieghts, f"cuda:{d}", batch_size)
            inferencer_params.append(inferencer)
        consumer_processes = []
        for _, param in zip(self.devices, inferencer_params):
            consumer_process = mp.Process(target=self._pred, args=param)
            consumer_processes.append(consumer_process)
            consumer_process.start()
        return consumer_processes
    
    def start_crop_process(self, wsi_path):
        consumer_processes = []
        for _ in range(self.crop_workers):
            consumer_process = mp.Process(target=self._crop, args=[wsi_path])
            consumer_processes.append(consumer_process)
            consumer_process.start()
            
        return consumer_processes
    
    def start_merge_process(self, rev_num, save_path):
        consumer_processes = []
        for _ in range(1):
            consumer_process = mp.Process(target=self._merge, args=[rev_num, save_path])
            consumer_processes.append(consumer_process)
            consumer_process.start()
            
        return consumer_processes
    
    def _merge(self, rev_num, save_path):
        
        all_preds = []
        total = 0
        progress = tqdm.tqdm(range(rev_num), total=rev_num)
        
        while True:
            try:
                pred = self.pred_queue.get()
                if pred is not None:
                    all_preds.append(pred)
                total += 1
                progress.update()
                if total >= rev_num:
                    break
            except Exception as e:
                print("Merge:", e)
                
        print("Merge: finished")
        print("Save preds ...")
        start = time.time_ns()

        with open(save_path, "w") as f:
            json.dump(all_preds, f)
            
        cost = (time.time_ns() - start) / 1e6
        print(f"Saved preds, cost {cost:.3f}ms")
        # print(self.region_queue.qsize(), self.image_queue.qsize())
        for _ in range(self.crop_workers):
            self.region_queue.put(None)    
        for _ in self.devices:
            self.image_queue.put(None)
        
        print("Done")
        
    def _pred(self, configs, wieghts, devices=0, batch_size=1):
        inferencer = DetInferencer(configs, weights=wieghts, device=devices, show_progress=False)
        
        while True:
            try:
                obj = self.image_queue.get()
                # print("Pred", obj)
                if obj is None:
                    break
                img, rg = obj
                if img is None:
                    self.pred_queue.put([None, rg])
                    continue
                x, y = rg
                res = inferencer(img, pred_score_thr=0.1)
                
                if not os.path.exists("debug.jpg"):
                    imageio.imwrite("debug.jpg", img)
                
                # print(res)
                
                predictions = res['predictions'][0]
                
                bboxes = np.array(predictions['bboxes']) #[N,4] xyxy

                bboxes[:, 0::2] += x
                bboxes[:, 1::2] += y
                predictions['bboxes'] = bboxes.tolist()
                
                self.pred_queue.put([predictions, rg])
            except Exception as e:
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                print("Predict:", e)
                break
        
        print(f"Predict Process {mp.current_process().name} stopped.")    
    
    def _crop(self, wsi_path):
        wsi = SdpcReader(wsi_path, {
            "crop_pixel_size": 0.2,
            "crop_size_h": 1024,
            "crop_size_w": 1024,
            "crop_overlap": 0
        })
        wsi.get_crop_region()
        while True:
            try:
                region = self.region_queue.get()
                
                if region is None:
                    break
                # print("Crop", "region", region)
                imgs = wsi.crop_patch(*region)
                
                for img_id, img in imgs:
                    # if img is not None:
                    #     print(img.shape)
                    # print("Start put", img)
                    x, y = list(map(int, img_id[:-4].split("_")[1:]))
                    self.image_queue.put([img, [x, y]])
                    # print("End put")
                    
            except Exception as e:
                print("Crop:", e)
                break
        wsi.close()
        print(f"Crop Process {mp.current_process().name} stopped.")
         
    
    def statistic(self, pred_json, wsi_size, region_size=[1024, 1024]):
        
        stem = Path(pred_json).stem
        if not os.path.exists(pred_json):
            raise RuntimeError("Nothing to statistic.")
        
        wsi_rg_size = [wsi_size[0] // region_size[0], wsi_size[1] // region_size[1]]
        heatmap = np.zeros(wsi_rg_size, dtype=np.int32) - 1
        with open(pred_json) as f:
            json_data = json.load(f)
            for img_pred, region in json_data:
                
                if img_pred:
                    labels = np.array(img_pred["labels"])
                    scores = np.array(img_pred["scores"])
                    bboxes = np.array(img_pred["bboxes"])
                
                    valide_idx = (labels == 0) & (scores > 0.5)
                    valide_bboxes = bboxes[valide_idx]
                else:
                    valide_bboxes = []
                    
                heatmap[region[0] // region_size[0], region[1] // region_size[1]] = len(valide_bboxes)
        
        hist = heatmap[heatmap > -1].flatten()
        print(f"[{Path(pred_json).stem}]: wsi patch {len(hist)}")
        print(hist.sum() / len(hist) * (78 / 4))
        # plt.hist(hist, bins=np.arange(-0.5, 20), log=True)
        labels, counts = np.unique(hist, return_counts=True)
        plt.bar(labels, counts, align='center')
        plt.gca().set_xticks(labels)
        plt.savefig(f"{stem}_hist.png")
        np.save(f"{stem}_Heatmap_Sampled_{len(hist)}.npy", heatmap)
        # heatmap[heatmap < 0] = -10
        # heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        # plt.imsave(f"{stem}_Heatmap_Sampled_{len(hist)}.png", heatmap, )
        
        # from scipy.interpolate import griddata
        # heatmap[heatmap < 0] = 0
        # r1, c1 = np.nonzero(heatmap)
        # values = heatmap[r1, c1]
        # r2 = np.arange(heatmap.shape[0])
        # c2 = np.arange(heatmap.shape[1])
        # data_list = np.meshgrid(r2, c2)
        # r3, c3 = data_list
        # point = np.vstack((r1, c1)).T
        
        # x5 = griddata(point, values, (r3, c3), method='linear')
        # x5 = x5.T
        # where_are_nan = np.isnan(x5)
        # x5[where_are_nan] = 0
        # mean1 = np.mean(x5)
        # x5[where_are_nan] = mean1
        
        # plt.imsave(f"{stem}_Heatmap_Full.png", x5, )
        # plt.close("all")

        
    def get_wsi_size(self, wsi_path):
        wsi = SdpcReader(wsi_path, {
            "crop_pixel_size": 0.2,
            "crop_size_h": 1024,
            "crop_size_w": 1024,
            "crop_overlap": 0
        })
        w, h = wsi.width, wsi.height
        wsi.close()
        return w, h
    
    def __call__(self, wsi_path, save_path) -> Any:
        # os.makedirs(Path(save_path).parent.__str__(), exist_ok=True)
        
        wsi = SdpcReader(wsi_path, {
            "crop_pixel_size": 0.2,
            "crop_size_h": 1024,
            "crop_size_w": 1024,
            "crop_overlap": 0
        })
        crop_processes, pred_processes = None, None
        try:

            regions = wsi.get_crop_region()
            w, h = wsi.width, wsi.height
            crop_h, crop_w = wsi.crop_size_h_, wsi.crop_size_w_
            print(w, h, wsi.slide_pixel_size)
            
            index = np.random.choice(regions.shape[0], min(2000, regions.shape[0]), replace=False)  
            regions = regions[index]
            rg_size = len(regions) * 4
            wsi.close()
            crop_processes = self.start_crop_process(wsi_path)
            pred_processes = self.start_pred_process(self.configs, self.wieghts, self.devices)
            merge_process = self.start_merge_process(rg_size, save_path)[0]
            
            print("Start...")
            cnt = 0
            for r in regions:
                self.region_queue.put(r)
                cnt+=1
            merge_process.join()
            self.statistic(save_path, wsi_size=[w, h], region_size=[crop_w, crop_h])
            
        except Exception as e:
            print(e)
            self.pred_queue.close()
            self.image_queue.close()
            self.region_queue.close()
            if crop_processes:
                for i in crop_processes:
                    i.join()
                    i.close()
            if pred_processes:
                for i in pred_processes:
                    i.close()
                
        
        
            
        
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    wsi_predictor = WSIPredictor(
            configs="/nasdata2/private/zwlu/detection/TumorBuddingDetection/mmdetection/work_dirs/rtmdet_l_swin_b_p6_4xb16_8000i_tb_v5_semi_mixpl_p_20_finetune/rtmdet_l_swin_b_p6_4xb16_200e_tb_v5_semi_mixpl.py", 
            # wieghts="/nasdata2/private/zwlu/detection/TumorBuddingDetection/mmdetection/work_dirs/rtmdet_l_swin_b_p6_4xb16_8000i_tb_v5_semi_mixpl_p_20_finetune/iter_6400.pth"
            wieghts="/nasdata2/private/zwlu/detection/TumorBuddingDetection/mmdetection/work_dirs/rtmdet_l_swin_b_p6_4xb16-100e_tb_v6_finetune/epoch_80.pth"
        )
    
    wsi_list = glob.glob("/nasdata/dataset/BD_testset/*.sdpc")
    
    for wsi_path in wsi_list:
        wsi_name = Path(wsi_path).stem
        wsi_predictor(wsi_path, f"wsi_outputs_v6/{wsi_name}.json")
    
    # wsi_predictor.statistic("/nasdata2/private/zwlu/detection/TumorBuddingDetection/mmdetection/wsi_outputs/2023-45-BD3.json", wsi_predictor.get_wsi_size("/nasdata/dataset/BD_testset/2023-45-BD3.sdpc"))
        
        
        
        
        
        
        


