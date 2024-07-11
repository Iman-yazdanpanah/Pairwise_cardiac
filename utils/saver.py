import os
import shutil
import torch
from collections import OrderedDict
import glob
import numpy as np
from PIL import Image

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkname)
        self.runs = glob.glob(os.path.join(self.directory, 'experiment_*'))
        run_id = sorted([int(self.runs[i].split('_')[-1]) for i in range(len(self.runs))])
        run_id = run_id[-1]+1 if run_id else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
            
        self.save_dir = os.path.join(self.experiment_dir, 'predict_mask')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['datset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['out_stride'] = self.args.out_stride
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['batch_size'] = self.args.batch_size
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()
        
    

    def save_predict_mask(self, predicts, image_indices, data1_files):
        """saves predicted mask to disk"""
        num_predictions = len(predicts)
        for i in range(min(len(image_indices), num_predictions)):
            image_index = image_indices[i]
            # Extract the file path from data1_files
            file_path = data1_files[image_index]
            
            # Ensure file_path is a string
            if not isinstance(file_path, str):
                raise TypeError(f"Expected file_path to be a string, but got {type(file_path).__name__}")
            
            _, name = os.path.split(file_path)
            name = name.split('.')[0]
            
            # Save as .npy file
            np.save(os.path.join(self.save_dir, name + '_output.npy'), predicts[i].astype(np.float16))
            
            # Save as PNG image
            prediction = predicts[i]  # Already a numpy array
            
            # Debugging: Print out some details about the prediction
            #print(f"Prediction {i}: min={prediction.min()}, max={prediction.max()}, shape={prediction.shape}")
            
            if prediction.ndim == 3 and prediction.shape[0] == 1:  # Assuming single-channel image
                prediction = prediction.squeeze(0)
            
            if prediction.min() < 0 or prediction.max() > 1:
                #print(f"Scaling prediction {i} from range ({prediction.min()}, {prediction.max()}) to (0, 255)")
                prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
            
            prediction = (prediction * 255).astype(np.uint8)  # Convert to 8-bit grayscale
            
            # Debugging: Print out the new min and max values after scaling
            #print(f"Prediction {i} after scaling: min={prediction.min()}, max={prediction.max()}, shape={prediction.shape}")
            
            # Create a PIL image
            img = Image.fromarray(prediction)
            
            # Save the image
            save_path = os.path.join(self.save_dir, name + '_output.png')
            img.save(save_path)
            #print(f"Saved prediction {i} as {save_path}")