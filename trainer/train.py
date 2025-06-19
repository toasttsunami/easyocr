import os
import sys
import time
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import csv
import json

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test_module import validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    print("Modules, Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        #table.add_row([name, param])
        total_params+=param
        print(name, param)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def initialize_metrics_logging(experiment_name):
    """Initialize CSV files for metrics logging"""
    metrics_dir = f'./saved_models/{experiment_name}/metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Training metrics CSV
    train_csv_path = os.path.join(metrics_dir, 'training_metrics.csv')
    train_fieldnames = ['iteration', 'train_loss', 'learning_rate', 'batch_time', 'data_time']
    
    # Validation metrics CSV
    val_csv_path = os.path.join(metrics_dir, 'validation_metrics.csv')
    val_fieldnames = ['iteration', 'valid_loss', 'accuracy', 'norm_ed', 'inference_time', 
                      'elapsed_time', 'best_accuracy', 'best_norm_ed']
    
    # Initialize CSV files with headers
    with open(train_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=train_fieldnames)
        writer.writeheader()
    
    with open(val_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=val_fieldnames)
        writer.writeheader()
    
    return train_csv_path, val_csv_path, train_fieldnames, val_fieldnames

def log_training_metrics(csv_path, fieldnames, metrics):
    """Log training metrics to CSV"""
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(metrics)

def log_validation_metrics(csv_path, fieldnames, metrics):
    """Log validation metrics to CSV"""
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(metrics)

def save_metrics_summary(experiment_name, metrics_summary):
    """Save overall training summary as JSON"""
    summary_path = f'./saved_models/{experiment_name}/metrics/training_summary.json'
    # Just before json.dump(metrics_summary, f, indent=2)
    for k, v in metrics_summary.items():
        if isinstance(v, torch.Tensor):
            metrics_summary[k] = v.item()
        elif isinstance(v, np.generic):  # for NumPy floats/ints
            metrics_summary[k] = v.item()
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2)

def train(opt, show_number = 2, amp=False):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.experiment_name}/log_dataset.txt', 'a', encoding="utf8")
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust=opt.contrast_adjust)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=min(32, opt.batch_size),
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers), prefetch_factor=512,
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    # Initialize metrics logging
    train_csv_path, val_csv_path, train_fieldnames, val_fieldnames = initialize_metrics_logging(opt.experiment_name)
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    if opt.saved_model != '':
        pretrained_dict = torch.load(opt.saved_model)
        if opt.new_prediction:
            model.Prediction = nn.Linear(model.SequenceModeling_output, len(pretrained_dict['module.Prediction.weight']))  
        
        model = torch.nn.DataParallel(model).to(device) 
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(pretrained_dict, strict=False)
        else:
            model.load_state_dict(pretrained_dict)
        if opt.new_prediction:
            model.module.Prediction = nn.Linear(model.module.SequenceModeling_output, opt.num_class)  
            for name, param in model.module.Prediction.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            model = model.to(device) 
    else:
        # weight initialization
        for name, param in model.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue
        model = torch.nn.DataParallel(model).to(device)
    
    model.train() 
    print("Model:")
    print(model)
    count_parameters(model)
    
    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # freeze some layers
    try:
        if opt.freeze_FeatureFxtraction:
            for param in model.module.FeatureExtraction.parameters():
                param.requires_grad = False
        if opt.freeze_SequenceModeling:
            for param in model.module.SequenceModeling.parameters():
                param.requires_grad = False
    except:
        pass
    
    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.optim=='adam':
        #optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizer = optim.Adam(filtered_parameters)
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.experiment_name}/opt.txt', 'a', encoding="utf8") as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    i = start_iter

    scaler = GradScaler()
    t1= time.time()
    
    # Metrics tracking
    training_losses = []
    validation_accuracies = []
    validation_losses = []
    learning_rates = []

    with tqdm(total=opt.num_iter, initial=start_iter, desc="Training Progress") as pbar:
        while(True):
            # Data loading time tracking
            data_start_time = time.time()
            
            # train part
            optimizer.zero_grad(set_to_none=True)
            
            if amp:
                with autocast():
                    image_tensors, labels = train_dataset.get_batch()
                    data_time = time.time() - data_start_time
                    
                    batch_start_time = time.time()
                    image = image_tensors.to(device)
                    text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
                    batch_size = image.size(0)
    
                    if 'CTC' in opt.Prediction:
                        preds = model(image, text).log_softmax(2)
                        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                        preds = preds.permute(1, 0, 2)
                        torch.backends.cudnn.enabled = False
                        cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                        torch.backends.cudnn.enabled = True
                    else:
                        preds = model(image, text[:, :-1])  # align with Attention.forward
                        target = text[:, 1:]  # without [GO] Symbol
                        cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
                scaler.scale(cost).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                image_tensors, labels = train_dataset.get_batch()
                data_time = time.time() - data_start_time
                
                batch_start_time = time.time()
                image = image_tensors.to(device)
                text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
                batch_size = image.size(0)
                if 'CTC' in opt.Prediction:
                    preds = model(image, text).log_softmax(2)
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    preds = preds.permute(1, 0, 2)
                    torch.backends.cudnn.enabled = False
                    cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                    torch.backends.cudnn.enabled = True
                else:
                    preds = model(image, text[:, :-1])  # align with Attention.forward
                    target = text[:, 1:]  # without [GO] Symbol
                    cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
                cost.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip) 
                optimizer.step()
            
            batch_time = time.time() - batch_start_time
            
            loss_avg.add(cost)
            training_losses.append(cost.item())
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else opt.lr
            learning_rates.append(current_lr)
            
            # Log training metrics every iteration
            train_metrics = {
                'iteration': i,
                'train_loss': cost.item(),
                'learning_rate': current_lr,
                'batch_time': batch_time,
                'data_time': data_time
            }
            log_training_metrics(train_csv_path, train_fieldnames, train_metrics)
            
            pbar.update(1)
            pbar.set_description(f"Training Progress (loss: {cost.item():.4f}, lr: {current_lr:.6f})")
            
            # validation part
            if (i % opt.valInterval == 0) and (i!=0):
                print('training time: ', time.time()-t1)
                t1=time.time()
                elapsed_time = time.time() - start_time
                # for log
                with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a', encoding="utf8") as log:
                    model.eval()
                    with torch.no_grad():
                        valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels,\
                        infer_time, length_of_data = validation(model, criterion, valid_loader, converter, opt, device)
                    model.train()
                    
                    # Store validation metrics
                    validation_losses.append(valid_loss)
                    validation_accuracies.append(current_accuracy)
    
                    # training loss and validation loss
                    loss_log = f'[{i}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                    loss_avg.reset()
    
                    current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}'
    
                    # keep best accuracy model (on valid dataset)
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
                    if current_norm_ED > best_norm_ED:
                        best_norm_ED = current_norm_ED
                        torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')
                    best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.4f}'
    
                    loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                    print(loss_model_log)
                    log.write(loss_model_log + '\n')
                    
                    # Log validation metrics
                    val_metrics = {
                        'iteration': i,
                        'valid_loss': valid_loss,
                        'accuracy': current_accuracy,
                        'norm_ed': current_norm_ED,
                        'inference_time': infer_time,
                        'elapsed_time': elapsed_time,
                        'best_accuracy': best_accuracy,
                        'best_norm_ed': best_norm_ED
                    }
                    log_validation_metrics(val_csv_path, val_fieldnames, val_metrics)
    
                    # show some predicted results
                    dashed_line = '-' * 80
                    head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                    predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                    
                    #show_number = min(show_number, len(labels))
                    
                    start = random.randint(0,len(labels) - show_number )    
                    for gt, pred, confidence in zip(labels[start:start+show_number], preds[start:start+show_number], confidence_score[start:start+show_number]):
                        if 'Attn' in opt.Prediction:
                            gt = gt[:gt.find('[s]')]
                            pred = pred[:pred.find('[s]')]
    
                        predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                    predicted_result_log += f'{dashed_line}'
                    print(predicted_result_log)
                    log.write(predicted_result_log + '\n')
                    print('validation time: ', time.time()-t1)
                    t1=time.time()
                    
            # save model per 1e+4 iter.
            if (i + 1) % 2e+3 == 0:
                torch.save(
                    model.state_dict(), f'./saved_models/{opt.experiment_name}/iter_{i+1}.pth')
                
                # Save intermediate metrics summary
                metrics_summary = {
                    'iteration': int(i + 1),
                    'total_iterations': int(opt.num_iter),
                    'best_accuracy': float(best_accuracy),
                    'best_norm_ed': float(best_norm_ED),
                    'current_train_loss': float(training_losses[-1]) if training_losses else 0.0,
                    'current_val_loss': float(validation_losses[-1]) if validation_losses else 0.0,
                    'current_val_accuracy': float(validation_accuracies[-1]) if validation_accuracies else 0.0,
                    'training_time_elapsed': float(time.time() - start_time),
                    'avg_train_loss_last_1000': float(np.mean(training_losses[-1000:])) if len(training_losses) >= 1000 else float(np.mean(training_losses)) if training_losses else 0.0,
                    'experiment_name': str(opt.experiment_name)
                }
                save_metrics_summary(opt.experiment_name, metrics_summary)
    
            if i == opt.num_iter:
                print('end the training')
                
                # Save final metrics summary
                final_summary = {
                    'total_iterations': int(opt.num_iter),
                    'final_best_accuracy': float(best_accuracy),
                    'final_best_norm_ed': float(best_norm_ED),
                    'total_training_time': float(time.time() - start_time),
                    'avg_train_loss': float(np.mean(training_losses)) if training_losses else 0.0,
                    'final_train_loss': float(training_losses[-1]) if training_losses else 0.0,
                    'final_val_loss': float(validation_losses[-1]) if validation_losses else 0.0,
                    'final_val_accuracy': float(validation_accuracies[-1]) if validation_accuracies else 0.0,
                    'experiment_name': str(opt.experiment_name),
                    'model_parameters': int(sum(params_num)),
                    'total_training_samples': int(len(training_losses))
                }                
                
                save_metrics_summary(opt.experiment_name, final_summary)
                
                print(f"Training completed! Metrics saved in: ./saved_models/{opt.experiment_name}/metrics/")
                print(f"Final accuracy: {best_accuracy:.3f}")
                print(f"Final norm ED: {best_norm_ED:.4f}")
                
                sys.exit()
            i += 1
