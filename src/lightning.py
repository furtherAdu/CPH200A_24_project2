import functools
import operator
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR
import torchmetrics
import torchmetrics.classification
import torchvision.models as models
from src.cindex import concordance_index
from einops import rearrange
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.models.video import swin3d_b, Swin3D_B_Weights
import os
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.metrics import RocCurveDisplay, roc_curve
from NLST_data_dict import subgroup_dict
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from einops.layers.torch import Reduce
from NLST_data_dict import clinical_feature_type, subgroup_feature_type
import os
import matplotlib.pyplot as plt
from einops import repeat

dirname = os.path.dirname(__file__)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def compute_iou(pred_mask, true_mask, threshold=0.5):
    pred_binary = (pred_mask > threshold).float()
    intersection = (pred_binary * true_mask).sum()
    union = pred_binary.sum() + true_mask.sum() - intersection
    if union == 0:
        return 0.0
    else:
        return (intersection / union).item()

def visualize_and_save_predictions(input_image, true_mask, pred_mask, patient_id, batch_idx, output_dir='visualizations/train'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # sample_dir = os.path.join(output_dir, f'patient_{patient_id}')
    # os.makedirs(sample_dir, exist_ok=True)

    input_image = input_image.cpu()
    true_mask = true_mask.cpu()
    pred_mask = pred_mask.cpu()

    pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, D', H', W']
    pred_mask_upsampled = F.interpolate(
        pred_mask,
        size=input_image.shape[1:],  # (D, H, W)
        mode='trilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)  # Shape: [D, H, W]

    input_image_np = input_image.numpy()
    true_mask_np = true_mask.numpy()
    pred_mask_np = pred_mask_upsampled.detach().numpy()

    non_zero_slices = np.where(true_mask_np.sum(axis=(1,2)) > 0)[0]

    if len(non_zero_slices) == 0:
        # print(f"Patient {patient_id} has no slices with non-zero mask.")
        return

    middle_idx = len(non_zero_slices) // 2
    slice_idx = non_zero_slices[middle_idx]

    # Handle input_image
    image_slice = input_image_np[:, slice_idx, :, :].mean(axis=0)  # Shape: [H, W]

    # Handle true_mask
    true_mask_slice = true_mask_np[slice_idx, :, :]  # Shape: [H, W]

    # Handle pred_mask
    pred_mask_slice = pred_mask_np[slice_idx, :, :]  # Shape: [H, W]

    # Normalize the attention map for visualization
    max_value = pred_mask_slice.max()
    if max_value > 0:
        normalized_pred_mask = pred_mask_slice / max_value
    else:
        normalized_pred_mask = pred_mask_slice  # Handle the case where max_value is zero

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Display input image
    axs[0].imshow(image_slice, cmap='gray')
    axs[0].set_title('Input Image', pad=10)  
    axs[0].axis('off')

    # Display ground truth mask
    axs[1].imshow(image_slice, cmap='gray')
    axs[1].imshow(true_mask_slice, alpha=0.5, cmap='Reds')
    axs[1].set_title('Ground Truth Mask', pad=10)  
    axs[1].axis('off')

    # Display attention heatmap
    axs[2].imshow(image_slice, cmap='gray')
    axs[2].imshow(normalized_pred_mask, alpha=0.5, cmap='jet')
    axs[2].set_title('Attention Heatmap', pad=10)  
    axs[2].axis('off')

    # Adjust layout and save the figure
    plt.tight_layout(pad=2)  
    output_path = os.path.join(output_dir, f'batch_{batch_idx}_pid_{patient_id}_slice_{slice_idx}.png')
    plt.savefig(output_path)
    plt.close()

class Classifer(pl.LightningModule):
    def __init__(self, num_classes=9, init_lr=3e-4):
        super().__init__()
        self.init_lr = init_lr
        self.num_classes = num_classes

        # Define loss fn for classifier
        self.loss = nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.auc = torchmetrics.AUROC(task="binary" if self.num_classes == 2 else "multiclass", num_classes=self.num_classes)
        self.confmat = torchmetrics.ConfusionMatrix(task="binary" if self.num_classes == 2 else "multiclass", num_classes=self.num_classes)

        self.training_outputs = []
        self.validation_outputs = []
        self.test_outputs = []

    def get_xy(self, batch, return_region_mask=False):
        if isinstance(batch, list):
            x, y = batch[0], batch[1]
        else:
            assert isinstance(batch, dict)
            x, y = batch["x"], batch["y_seq"][:,0]
        
        if return_region_mask:
            region_mask = batch['mask']
            return x, y.to(torch.long).view(-1), region_mask
        else:
            return x, y.to(torch.long).view(-1)

    def training_step(self, batch, batch_idx):
        x, y = self.get_xy(batch)

        # get predictions from your model and store them as y_hat
        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)

        ## Store the predictions and labels for use at the end of the epoch
        self.training_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.get_xy(batch)

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)

        self.validation_outputs.append({
            "y_hat": y_hat,
            "y": y
        })

        if self.trainer.datamodule.name == 'NLST':
            self.validation_outputs[-1].update({
                            "criteria": batch[self.trainer.datamodule.criteria],
                            **{k:batch[k] for k in self.trainer.datamodule.group_keys},
            })
        return loss

    def test_step(self, batch, batch_idx):
        x, y = self.get_xy(batch)

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        self.log('test_loss', loss, sync_dist=True, prog_bar=True)
        self.log('test_acc', self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)

        self.test_outputs.append({
            "y_hat": y_hat,
            "y": y
        })

        if self.trainer.datamodule.name == 'NLST':
            self.test_outputs[-1].update({
                             "criteria": batch[self.trainer.datamodule.criteria],
                            **{k:batch[k] for k in self.trainer.datamodule.group_keys},
            })
        return loss
    
    def on_train_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.training_outputs])
        y = torch.cat([o["y"] for o in self.training_outputs])
        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)
        self.log("train_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)

        # confusion matrix metrics for model predictions
        conf_metrics = self.get_confmat_metrics(probs if self.num_classes==2 else probs.argmax(dim=1), y)
        self.log_dict({f'train_{k}': v for k,v in conf_metrics.items()}, sync_dist=True, prog_bar=True)

        self.training_outputs = []

    def on_validation_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])
        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)
        self.log("val_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)

        # confusion matrix metrics for model predictions
        conf_metrics = self.get_confmat_metrics(probs if self.num_classes==2 else probs.argmax(dim=1), y)
        self.log_dict({f'val_{k}':v for k,v in conf_metrics.items()}, sync_dist=True, prog_bar=True)

        # confusion matrix metrics for criteria (e.g. lungrads)
        if self.trainer.datamodule.name == 'NLST':
            criteria = torch.cat([o["criteria"] for o in self.validation_outputs])
            criteria_conf_metrics = self.get_confmat_metrics(criteria.squeeze(), y.squeeze())
            self.log_dict({f'val_{k}_{self.trainer.datamodule.criteria}':v 
                           for k,v in criteria_conf_metrics.items()}, sync_dist=True, prog_bar=True)

        self.roc_analysis_across_nodes(self.validation_outputs, plot_label='val set')
        self.validation_outputs = []

    def on_test_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.test_outputs])
        y = torch.cat([o["y"] for o in self.test_outputs])

        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)

        self.log("test_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)

        # confusion matrix metrics for model prediction
        conf_metrics = self.get_confmat_metrics(probs if self.num_classes==2 else probs.argmax(dim=1), y)
        self.log_dict({f'test_{k}':v for k,v in conf_metrics.items()}, sync_dist=True, prog_bar=True)

        # confusion matrix metrics for criteria (e.g. lungrads)
        if self.trainer.datamodule.name == 'NLST':
            criteria = torch.cat([o['criteria'] for o in self.test_outputs])
            criteria_conf_metrics = self.get_confmat_metrics(criteria.squeeze(), y.squeeze())
            self.log_dict({f'test_{k}_{self.trainer.datamodule.criteria}':v 
                           for k,v in criteria_conf_metrics.items()}, sync_dist=True, prog_bar=True)            


        self.roc_analysis_across_nodes(self.test_outputs, plot_label='test set')
        self.test_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        scheduler = LinearLR(optimizer)
        return [optimizer], [scheduler]
    
    def init_weights(m, nonlinearity='relu'):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)

    def roc_analysis_across_nodes(self, split_outputs, plot_label='val set'):
        if self.trainer.datamodule.name == 'NLST':
            output_across_samples = self.get_outputs_across_samples(split_outputs)
            
            # init empty tensors to gather tensors across nodes
            output_across_nodes = {}
            for k,v in output_across_samples.items():
                node_v_size = list(v.size())
                node_v_size[0] *= self.trainer.world_size
                output_across_nodes[k] = torch.zeros(node_v_size).type(v.type()).to(self.device).contiguous()

            # wait for all nodes to catch up
            torch.distributed.barrier()

            # gather each tensor necessary for ROC plotting
            for name, tensor in output_across_samples.items():
                torch.distributed.all_gather_into_tensor(output_across_nodes[name], tensor.contiguous(), async_op=False)

            if self.global_rank == 0: # on node 0
                # send to numpy
                output_across_nodes = {k:self.safely_to_numpy(v).squeeze() for k,v in output_across_nodes.items()}

                # transform arrays as needed
                for k in self.trainer.datamodule.group_keys:
                    feature_type = subgroup_feature_type[k]
                    if k in self.trainer.datamodule.vectorizer.features_fit[feature_type]:
                        feature = self.trainer.datamodule.vectorizer.transform({k: output_across_nodes[k]}, feature_type)
                        if feature_type == 'categorical':
                            feature[k] = np.argmax(feature[k], axis=1)
                        output_across_nodes.update(feature)
                    
                # print(probs, y.view(-1), group_data) # samples are 2x batch_size
                self.roc_analysis(output_across_nodes=output_across_nodes,
                                  plot_label=plot_label)
                
    def get_outputs_across_samples(self, split_outputs, output_keys=['y', 'y_hat', 'criteria']):
        # collect outputs across samples into a single tensor
        output_across_samples = {}
        for k in [*output_keys, *self.trainer.datamodule.group_keys]: 
            output = torch.cat([o[k] for o in split_outputs])
            if k == 'y_hat':
                if self.num_classes == 2:
                    output = F.softmax(output, dim=-1)[:,-1]
                else:
                    output = F.softmax(output, dim=-1)
            output = output.view(-1)
            output_across_samples.update({k: output})
        
        return output_across_samples

    @ staticmethod
    def safely_to_numpy(tensor):
        return tensor.to(torch.float).cpu().numpy()

    @staticmethod
    def plot_roc_operation_point(y, y_hat, ax, plot_label, color='g'):
        fpr, tpr, _ = roc_curve(y, y_hat)
        ax.plot(fpr[1], tpr[1], color=color, marker='o', linestyle='None', label=f'{plot_label} operation point')
    
    def roc_analysis(self, output_across_nodes, plot_label):
        # unpack data
        y = output_across_nodes['y']
        y_hat = output_across_nodes['y_hat']
        criteria = output_across_nodes['criteria']
        group_data = {k:v for k,v in output_across_nodes.items() if k in self.trainer.datamodule.group_keys}

        # generate plots
        roc_plot = self.generate_roc(y, y_hat, criteria)
        print('ROC curve generated.')
        if group_data:
            subgroup_roc_plot = self.generate_subgroup_roc(y, y_hat, criteria, group_data, plot_label=self.trainer.datamodule.criteria)
            print('Subgroup ROC curve generated.')
        
        # generate plot names
        plot_name = f'ROC, {plot_label}, epoch{self.current_epoch}'
        subgroup_plot_name = f'ROC by subgroups, {plot_label}, epoch{self.current_epoch}'
        
        # log plots
        if self.logger.experiment:
            wandb_logger = self.logger
            wandb_logger.log_image(key=plot_name, images=[roc_plot])
            if group_data:
                wandb_logger.log_image(key=subgroup_plot_name, images=[subgroup_roc_plot])

    def generate_roc(self, y, y_hat, criteria):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))

        self.plot_roc_operation_point(y,
                                      criteria,
                                      ax=ax,
                                      plot_label=self.trainer.datamodule.criteria)

        RocCurveDisplay.from_predictions(y,
                                         y_hat,
                                         name=None,
                                         plot_chance_level=True,
                                         ax=ax)

        ax.legend(loc='lower right', prop={'size': 8})
        plt.close()

        return fig

    def generate_subgroup_roc(self, y, y_hat, criteria, group_data, plot_label):
        # set up figure
        ncols = 2
        nrows = ceil(len(group_data)/ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(int(ncols*10), (int(nrows*8))))
        axs = axs.ravel()
        axs_count = 0
        for group_key, group_i in group_data.items(): 
            # Plot operation point for_criteria
            self.plot_roc_operation_point(y,
                                          criteria,
                                          ax=axs[axs_count],
                                          plot_label=plot_label)
            
            subgroups = np.unique(group_i)
            for j, subgroup in enumerate(subgroups):
                # get group indices
                subgroup_idxs = np.argwhere(group_i == subgroup)

                # get subgroup name
                feature_type = subgroup_feature_type[group_key]
                if group_key in self.trainer.datamodule.vectorizer.features_fit[feature_type]:
                    subgroup_name = self.trainer.datamodule.vectorizer.feature_levels[feature_type][group_key][j]
                else:
                    subgroup_name = subgroup_dict[group_key][subgroup]                    

                # get roc curve kwargs
                roc_kwargs = dict(name=subgroup_name, 
                                  pos_label=1,
                                  ax=axs[axs_count])
                
                if j == len(subgroups) - 1:
                    roc_kwargs.update(dict(plot_chance_level=True))

                # generate roc curve
                RocCurveDisplay.from_predictions(y[subgroup_idxs],
                                                 y_hat[subgroup_idxs], 
                                                 **roc_kwargs)
            
            axs[axs_count].grid()
            axs[axs_count].legend(loc='lower right', prop={'size': 8})
            axs_count += 1
        
        # remove unused subplots
        for i in range(axs_count, int(nrows*ncols)):
            fig.delaxes(axs[i])

        plt.tight_layout()
        plt.close()

        return fig        

    def get_confmat_metrics(self, y_hat, y, by_class=False):
        # Calculate confusion matrix 
        cm = self.confmat(y_hat, y) # axes: (true, predicted)

        # get pos/neg count per class
        TP = cm.diag()
        FP = cm.sum(dim=0) - TP
        FN = cm.sum(dim=1) - TP 
        TN = cm.sum() - (FP + FN + TP)
                
        metric_dict = {
            'sensitivity': TP/(TP+FN), # tpr
            'specificity': TN/(TN+FN), # tnr
            'precision': TP/(TP+FP), # ppv
            'fpr': FP/(FP+TN),
        }

        if not by_class:
            metric_dict = {k:v.mean() for k,v in metric_dict.items()}

        return metric_dict

    def step_3d(self, batch, batch_idx, stage, outputs, visualize_localization=False):
        x, y, region_mask = self.get_xy(batch, return_region_mask=True)

        y_hat, activation_scores = self.forward(x, return_maps=True)
        
        # Resize region mask to match activation scores size
        region_mask_resized = self.get_region_mask_resized(region_mask, activation_scores)
        

        # calculate losses
        use_localization = True
        classification_loss = self.get_classification_loss(y_hat, y)
        localization_loss = self.get_localization_loss(activation_scores, region_mask_resized) if use_localization else 0
        total_loss = classification_loss + 0.5 * localization_loss  # Adjust weight as needed

        batch_size = activation_scores.size(0)
        activation_scores_flat = activation_scores.view(batch_size, -1)
        alpha = F.softmax(activation_scores_flat, dim=1)
        alpha = alpha.view_as(activation_scores)
        iou = compute_iou(alpha, region_mask_resized)
        # # Compute metrics
        # iou = compute_iou(activation_map_normalized, region_mask_resized) if region_mask_resized.sum() > 0 else float('nan')
        accuracy = self.accuracy(y_hat, y)

        metric_dict = {'loss': total_loss, 
                       'acc': accuracy,
                       'classification_loss': classification_loss,
                       'localization_loss': localization_loss,
                       'iou': iou}
        
        # Log metrics to wandb
        for metric_name, metric_value in metric_dict.items():
            self.log(f'{stage}_{metric_name}', metric_value, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)
        
        outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        if self.trainer.datamodule.name == 'NLST':
            outputs[-1].update({
                "criteria": batch[self.trainer.datamodule.criteria],
                **{k:batch[k] for k in self.trainer.datamodule.group_keys},
            })

        # Visualization
        if visualize_localization:
            for sample_idx in range(x.size(0)):
                input_image = x[sample_idx]  # Shape: [C, D, H, W]
                true_mask = region_mask[sample_idx, 0]  # Assuming region_mask has shape [B, 1, D, H, W]
                pred_mask = alpha[sample_idx, 0]  # Shape: [D, H, W]

                # Get patient id
                patient_id = batch['pid'][sample_idx].item()

                # Call visualization function
                visualize_and_save_predictions(
                    input_image,
                    true_mask,
                    pred_mask,
                    patient_id=patient_id,
                    batch_idx=batch_idx,
                    output_dir=f'visualizations/{self.__class__.__name__}/{stage}'
                )

        return total_loss
    
    def get_activation_map_reduced_normalized(self, activation_map):
        # Reduce activation map to single channel
        activation_map_reduced = activation_map.mean(dim=1, keepdim=True)

        # Normalize using sigmoid
        activation_map_normalized = torch.sigmoid(activation_map_reduced)

        return activation_map_reduced, activation_map_normalized
    
    def get_region_mask_resized(self, region_mask, target_tensor):
        # Resize region mask to match the size of the target tensor (activation_scores)
        region_mask_resized = F.interpolate(
            region_mask,
            size=target_tensor.shape[2:],  # Match spatial dimensions (D', H', W')
            mode='nearest'  # Use nearest interpolation for masks
        ).float()

        return region_mask_resized

    # def get_localization_loss(self, activation_map, region_mask):
    #     localization_loss = self.localization_loss_fn(activation_map, region_mask)
    #     return localization_loss

    def get_localization_loss(self, activation_scores, region_mask_resized, epsilon=1e-8):
        batch_size = activation_scores.size(0)
        activation_scores_flat = activation_scores.view(batch_size, -1)  # Shape: [B, num_locations]
        region_mask_flat = region_mask_resized.view(batch_size, -1)      # Shape: [B, num_locations]

        # Apply softmax over spatial dimensions
        alpha = F.softmax(activation_scores_flat, dim=1)  # Shape: [B, num_locations]

        # Compute agreement
        agreement = torch.sum(alpha * region_mask_flat, dim=1)  # Shape: [B]

        # Compute localization loss
        localization_loss = -torch.log(agreement + epsilon)  # Add epsilon to prevent log(0)

        # Return mean loss over batch
        return localization_loss.mean()

    def get_classification_loss(self, y_hat, y):
        classification_loss = self.loss(y_hat, y)
        return classification_loss

class MLP(Classifer):
    def __init__(self, input_dim=28*28*3, hidden_dim=128, num_layers=1, num_classes=9, use_bn=False, init_lr=1e-3, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.use_bn = use_bn
        self.bn = [nn.BatchNorm1d(hidden_dim)] if self.use_bn else []
        self.num_layers = num_layers

        self.first_layer = nn.Sequential(nn.Linear(input_dim, self.hidden_dim),
                                         *self.bn,
                                         nn.ReLU())

        self.hidden_layers = []
        for _ in range(self.num_layers - 1):
            self.hidden_layers.append(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                    *self.bn,
                                                    nn.ReLU())
                                         )

        self.final_layer = nn.Sequential(nn.Linear(self.hidden_dim, num_classes),    
                                         nn.Softmax(dim=-1)
                                         )

        self.model = nn.Sequential(self.first_layer,
                                   *self.hidden_layers,
                                   self.final_layer
                                   )
        
        self.model.apply(self.init_weights)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = rearrange(x, 'b c w h -> b (w h c)')
        return self.model(x)


class LinearModel(Classifer):
    def __init__(self, input_dim=28*28*3, hidden_dim=128, num_layers=1, num_classes=9, use_bn=False, init_lr=1e-3, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.use_bn = use_bn
        self.bn = [nn.BatchNorm1d(self.hidden_dim)] if self.use_bn else []
        self.num_layers = num_layers

        self.first_layer = nn.Sequential(nn.Linear(input_dim, self.hidden_dim),
                                         *self.bn)

        self.hidden_layers = []
        for _ in range(self.num_layers - 1):
            self.hidden_layers.append(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                    *self.bn)
                                         )

        self.final_layer = nn.Sequential(nn.Linear(self.hidden_dim, num_classes),    
                                         nn.Softmax(dim=-1)
                                         )

        self.model = nn.Sequential(self.first_layer,
                                   *self.hidden_layers,
                                   self.final_layer
                                   )

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = rearrange(x, 'b c w h -> b (w h c)')
        return self.model(x)


class CNN(Classifer):
    def __init__(self, input_dim=(3, 28, 28), hidden_dim=128, num_layers=1, num_classes=9, use_bn=False, init_lr = 1e-3, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.use_bn = use_bn
        self.bn_fc = [nn.BatchNorm1d(hidden_dim)] if self.use_bn else []
        self.num_layers = num_layers

        # initialize convolutional layers 
        self.feature_extractor = []
        for i in range(num_layers):
            if i == 0: # first conv layer
                in_channels = input_dim[0]
                out_channels = 20
                k = 5 # Conv2d kernel size
                conv_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(k,k)),
                                           nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                           nn.ReLU()
                                            )
            else: # subsequent conv layers
                k = 3
                bn_conv = [nn.BatchNorm2d(out_channels)] if self.use_bn else []
                conv_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(k,k)),
                                           *bn_conv,
                                           nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                           nn.ReLU()
                                           )

            self.feature_extractor.append(conv_layer)
            
            # set channels for i > 0
            in_channels = out_channels
            out_channels *= 2
        
        self.feature_extractor = nn.Sequential(*self.feature_extractor)

        # get number of output features from conv layers
        num_features_before_fc = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, *input_dim)).shape))
            
		# initialize fully connected layers
        self.classifier = []
        for i in range(num_layers):
            in_features = num_features_before_fc if i == 0 else self.hidden_dim

            if i == num_layers - 1: # add final fc layer
                fc_layer = nn.Sequential(nn.Linear(in_features, num_classes),
                                         nn.Softmax(dim=-1)
                                         )
                        
            else: # add consequent fc layres
                fc_layer = nn.Sequential(nn.Linear(in_features=in_features, out_features=self.hidden_dim),
                                        *self.bn_fc,
                                        nn.ReLU()
                                        )
        
            self.classifier.append(fc_layer)
        
        self.classifier = nn.Sequential(*self.classifier)              

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = rearrange(x, 'b c w h -> b c h w')
        x = self.feature_extractor(x).flatten(1)
        return self.classifier(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

class CNN3D(Classifer):
    def __init__(self, input_dim=(3, 200, 128, 128), hidden_dim=128, num_layers=1, num_classes=2, use_bn=False, init_lr=1e-3, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.use_bn = use_bn
        self.bn_fc = [nn.BatchNorm1d(hidden_dim)] if self.use_bn else []
        self.num_layers = num_layers

        # Initialize convolutional layers
        self.feature_extractor = []
        for i in range(num_layers):
            if i == 0:  # First conv layer
                in_channels = input_dim[0]
                out_channels = 20
                k = 3  # Conv3d kernel size
                conv_layer = nn.Sequential(
                    nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(k, k, k)),
                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                    nn.ReLU()
                )
            else:  # Subsequent conv layers
                k = 3
                bn_conv = [nn.BatchNorm3d(out_channels)] if self.use_bn else []
                conv_layer = nn.Sequential(
                    nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(k, k, k)),
                    *bn_conv,
                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                    nn.ReLU()
                )

            self.feature_extractor.append(conv_layer)

            # Update channels for layers i > 0
            in_channels = out_channels
            out_channels *= 2

        self.feature_extractor = nn.Sequential(*self.feature_extractor)

        # Get the number of output features from conv layers
        num_features_before_fc = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, *input_dim)).shape[1:]))

        # Initialize fully connected layers
        self.classifier = []
        for i in range(num_layers):
            in_features = num_features_before_fc if i == 0 else self.hidden_dim

            if i == num_layers - 1:  # Final fc layer
                fc_layer = nn.Sequential(
                    nn.Linear(in_features, num_classes),
                    nn.Softmax(dim=-1)
                )
            else:  # Hidden fc layers
                fc_layer = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=self.hidden_dim),
                    *self.bn_fc,
                    nn.ReLU()
                )

            self.classifier.append(fc_layer)

        self.classifier = nn.Sequential(*self.classifier)

    def forward(self, x):
        # x shape: (batch_size, channels, depth, height, width)
        # Expand channels
        x = repeat(x, 'b c d h w -> b (repeat c) d h w', repeat=3)

        x = self.feature_extractor(x).flatten(1)
        return self.classifier(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
class ResNet18(Classifer):
    def __init__(self, num_classes=9, init_lr=1e-3, pretraining=False, depth_handling='max_pool', **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()
        
        # Initialize a ResNet18 model
        weights_kwargs = {'weights': models.ResNet18_Weights.DEFAULT} if pretraining else {} 
        self.backbone = models.resnet18(**weights_kwargs)        
        self.classification_head = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.backbone.fc = nn.Identity()
        self.pool = Reduce('b c h w -> b c 1 1', 'max') # max pooling
        
        # Add handling for depth dimension
        self.depth_handling = depth_handling
        
        if depth_handling == '3d_conv':
            # Replace first conv layer with 3D conv
            self.backbone.conv1 = nn.Conv3d(
                in_channels=3,
                out_channels=64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False
            )
            # Add a transition layer after first conv to go back to 2D
            self.transition = nn.Sequential(
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
            )
        
        if not pretraining:
            self.backbone.apply(self.init_weights)
            self.classification_head.apply(self.init_weights)

    def forward(self, x, return_logits=True, return_features=False, return_maps=False):
        # Expand channels
        x = repeat(x, 'b c d h w -> b (repeat c) d h w', repeat=3)

        # x shape: [batch_size, channels, depth, height, width]
        batch_size, channels, depth, height, width = x.size()
        
        if self.depth_handling == 'max_pool':
            # Method 1: Max pool across depth dimension
            x = x.max(dim=2)[0]  # Shape becomes [batch_size, channels, height, width]
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            activation_map = self.backbone.layer4(x)
            
        elif self.depth_handling == 'avg_pool':
            # Method 2: Average pool across depth dimension
            x = x.mean(dim=2)  # Shape becomes [batch_size, channels, height, width]
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            activation_map = self.backbone.layer4(x)
            
        elif self.depth_handling == 'slice_attention':
            # Method 3: Learn attention weights for each slice
            # First, reshape to treat depth as batch dimension
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(-1, channels, height, width)
            
            # Get features for each slice
            features = self.backbone.conv1(x)
            features = self.backbone.bn1(features)
            features = self.backbone.relu(features)
            features = self.backbone.maxpool(features)
            features = self.backbone.layer1(features)
            
            # Reshape back to include depth
            features = features.view(batch_size, depth, -1)
            
            # Simple attention mechanism
            attention_weights = F.softmax(self.backbone.avgpool(features), dim=1)
            weighted_features = (features * attention_weights).sum(dim=1)
            
            # Continue through rest of network
            x = self.backbone.layer2(weighted_features)
            x = self.backbone.layer3(x)
            activation_map = self.backbone.layer4(x)

        elif self.depth_handling == '3d_conv':
            # Method 4: Start with 3D convolutions
            x = self.backbone.conv1(x)
            x = self.transition(x)
            
            # Reshape to 2D after initial 3D conv
            x = x.squeeze(2)  # Remove depth dimension
            
            # Continue through rest of the network
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            activation_map = self.backbone.layer4(x)
        
        pooled_features = self.pool(activation_map).squeeze(dim=(2,3))

        # get objects to return
        return_objects = []
        if return_logits: # class prediction
            logits = self.classification_head(pooled_features)
            return_objects.append(logits)
        if return_features:
            return_objects.append(pooled_features)
        if return_maps:
            return_objects.append(activation_map)
                    
        return tuple(return_objects)

    def training_step(self, batch, batch_idx):
        return self.step_3d(batch, batch_idx, "train", self.training_outputs, visualize_localization=False)
    def validation_step(self, batch, batch_idx):
        return self.step_3d(batch, batch_idx, "val", self.validation_outputs)
    def test_step(self, batch, batch_idx):
        return self.step_3d(batch, batch_idx, "test", self.test_outputs)

class ResNet18_adapted(Classifer):
    def __init__(self, num_classes=2, init_lr=1e-4, pretraining=True):
        super().__init__(num_classes=num_classes, init_lr=init_lr)

        # Initialize ResNet18
        weights_kwargs = {"weights": models.ResNet18_Weights.DEFAULT} if pretraining else {}
        self.classifier = models.resnet18(**weights_kwargs)

        # Adjust the final layer for the number of classes
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)

        # Initialize weights if not pretraining
        if not pretraining:
            self.classifier.apply(self.init_weights)

    def forward(self, x):
        B, C, D, H, W = x.shape

        # Ensure input has 3 channels by repeating the channel dimension
        if C == 1:
            x = x.repeat(1, 3, 1, 1, 1)  # (B, 3, D, H, W)

        # Reshape and process each slice independently
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, 3, H, W)  # (B * D, 3, H, W)

        # Pass through ResNet18
        slice_preds = self.classifier(x)  # (B * D, num_classes)

        # Reshape back to (B, D, num_classes)
        slice_preds = slice_preds.view(B, D, self.num_classes)

        # Aggregate slice predictions
        volume_preds = slice_preds.mean(dim=1)  # (B, num_classes)

        return volume_preds

class ResNet3D(Classifer):
    def __init__(self, num_classes=2, init_lr=1e-3, pretraining=False, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        # Load the pretrained ResNet3D model
        weights_kwargs = {'weights': R3D_18_Weights.DEFAULT} if pretraining else {}
        self.backbone = r3d_18(**weights_kwargs)

        # Modify the model to remove the original classification head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the fully connected layer

        # Define pooling operation
        self.pool = Reduce('b c d h w -> b c 1 1 1', 'max') # max pooling

        # Define a new classification head
        self.classification_head = nn.Linear(num_features, num_classes)
        self.localization_head = nn.Conv3d(num_features, 1, kernel_size=1, bias=False)

    def forward(self, x, return_logits=True, return_features=False, return_maps=False):
        # Expand channels
        x = repeat(x, 'b c d h w -> b (repeat c) d h w', repeat=3)

        # Pass input through the backbone
        features = self.backbone.stem(x)
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        activation_map = self.backbone.layer4(features)  # Activation maps from the last conv layer
        activation_scores = self.localization_head(activation_map)  # Shape: [B, 1, D', H', W']

        # pooling
        pooled_features = self.pool(activation_map).squeeze(dim=(2,3,4))
        
        # get objects to return
        return_objects = []
        if return_logits: # class prediction
            logits = self.classification_head(pooled_features)
            return_objects.append(logits)
        if return_features:
            return_objects.append(pooled_features)
        if return_maps:
            return_objects.append(activation_scores)
                    
        return tuple(return_objects)

    def training_step(self, batch, batch_idx):
        return self.step_3d(batch, batch_idx, "train", self.training_outputs, visualize_localization=False)
    def validation_step(self, batch, batch_idx):
        return self.step_3d(batch, batch_idx, "val", self.validation_outputs)
    def test_step(self, batch, batch_idx):
        return self.step_3d(batch, batch_idx, "test", self.test_outputs)


class Swin3DModel(Classifer):
    def __init__(self, num_classes=2, init_lr=1e-3, pretraining=True, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        # Load the pretrained Swin3D model
        if pretraining:
            weights = Swin3D_B_Weights.DEFAULT
            self.backbone = swin3d_b(weights=weights)
        else:
            self.backbone = swin3d_b(weights=None)

        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()  # Remove the original head

        # Define pooling operation
        self.pool = Reduce('b c d h w -> b c 1 1 1', 'max') # max pooling

        # Define a new classification head
        self.classification_head = nn.Linear(in_features, num_classes)

    def forward(self, x, return_logits=True, return_features=False, return_maps=False):
        # Expand channels
        x = repeat(x, 'b c d h w -> b (repeat c) d h w', repeat=3)

        # Extract features using the backbone
        x = self.backbone.patch_embed(x)  # B _T _H _W C
        x = self.backbone.pos_drop(x)
        x = self.backbone.features(x)  # B _T _H _W C 
        x = self.backbone.norm(x)
        activation_map = x.permute(0, 4, 1, 2, 3)  # B, C, _T, _H, _W, Activation maps

        # pooling
        pooled_features = self.pool(activation_map).squeeze(dim=(2,3,4))

        # get objects to return
        return_objects = []
        if return_logits: # class prediction
            logits = self.classification_head(pooled_features)
            return_objects.append(logits)
        if return_features:
            return_objects.append(pooled_features)
        if return_maps:
            return_objects.append(activation_map)
                    
        return tuple(return_objects)
    
    def training_step(self, batch, batch_idx):
        return self.step_3d(batch, batch_idx, "train", self.training_outputs, visualize_localization=False)
    def validation_step(self, batch, batch_idx):
        return self.step_3d(batch, batch_idx, "val", self.validation_outputs)
    def test_step(self, batch, batch_idx):
        return self.step_3d(batch, batch_idx, "test", self.test_outputs)

NLST_CENSORING_DIST = {
    "0": 0.9851928130104401,
    "1": 0.9748317321074379,
    "2": 0.9659923988537479,
    "3": 0.9587252204657843,
    "4": 0.9523590830936284,
    "5": 0.9461840310101468,
}

class Cumulative_Probability_Layer(nn.Module):
    # adapted from: https://github.com/yala/Mirai/blob/master/onconet/models/cumulative_probability_layer.py
    def __init__(self, num_features, max_followup):
        super(Cumulative_Probability_Layer, self).__init__()

        # init model components
        self.hazard_fc = nn.Linear(num_features, max_followup) # hazard risk
        self.base_hazard_fc = nn.Linear(num_features, 1)  # baseline risk
        self.relu = nn.ReLU(inplace=True)

    def hazards(self, x):
        raw_hazard = self.hazard_fc(x)
        pos_hazard = self.relu(raw_hazard) # enforce positive hazard scores
        return pos_hazard

    def forward(self, x):
        hazards = self.hazards(x)
        cum_prob = torch.cumsum(hazards, dim=1) + self.base_hazard_fc(x)
        return cum_prob # logits

class RiskModel(Classifer):
    def __init__(self, num_classes=2, init_lr=1e-3, max_followup=6, backbone=None, clinical_features=[], **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters(ignore=['backbone'])
        self.max_followup = max_followup
        self.clinical_features = clinical_features

        # Use the modified ResNet3D model directly
        self.backbone = backbone  # Assume backbone is an instance of the modified ResNet3D

        # get number of input features
        num_features = backbone.classification_head.in_features + len(clinical_features)

        # Define classification head
        self.classification_head = Cumulative_Probability_Layer(num_features=num_features,
                                                                max_followup=self.max_followup)

        # Initialize metrics
        self.auc = torchmetrics.AUROC(task="binary")
        self.iou_metric = torchmetrics.JaccardIndex(task="binary")
        self.dice_metric = torchmetrics.Dice()
        self.accuracy = torchmetrics.classification.BinaryAccuracy()

    def forward(self, x, return_logits=True, return_features=False, return_maps=False, added_features=None):
        # Get logits and activation maps from the backbone
        pooled_features, activation_map = self.backbone(x, return_logits=False, return_features=True, return_maps=True)

        if added_features is not None:
            pooled_features = torch.cat([pooled_features, added_features], dim=1)

         # get objects to return
        return_objects = []
        if return_logits: # class prediction
            logits = self.classification_head(pooled_features)
            return_objects.append(logits)
        if return_features:
            return_objects.append(pooled_features)
        if return_maps:
            return_objects.append(activation_map)
                    
        return tuple(return_objects)

    def get_xy(self, batch):
        """
            x: (B, C, D, W, H) -  Tensor of CT volume
            y_seq: (B, T) - Tensor of cancer outcomes. a vector of [0,0,1,1,1, 1] means the patient got between years 2-3, so
            had cancer within 3 years, within 4, within 5, and within 6 years.
            y_mask: (B, T) - Tensor of mask indicating future time points are observed and not censored. For example, if y_seq = [0,0,0,0,0,0], then y_mask = [1,1,0,0,0,0], we only know that the patient did not have cancer within 2 years, but we don't know if they had cancer within 3 years or not.
            mask: (B, D, W, H) - Tensor of mask indicating which voxels are inside an annotated cancer region (1) or not (0).
                TODO: You can add more inputs here if you want to use them from the NLST dataloader.
                Hint: You may want to change the mask definition to suit your localization method

        """
        x = batch['x']
        y_seq = batch['y_seq'][:, :self.max_followup]
        y_mask = batch['y_mask'][:, :self.max_followup]

        region_annotation_mask = batch['mask']
        region_annotation_mask = (region_annotation_mask > 0).float()

        # # Debug statements
        # print(f"x shape: {x.shape}")  # Expected: (B, C, D, W, H)
        # print(f"y_seq shape: {y_seq.shape}")  # Expected: (B, T)
        # print(f"y_mask shape: {y_mask.shape}")  # Expected: (B, T)
        # print(f"region_annotation_mask shape: {region_annotation_mask.shape}")  # Expected: (B, D, W, H)")
        # print(f"region_annotation_mask unique values: {region_annotation_mask.unique()}")  # Should be [0, 1]
        
        return x, y_seq, y_mask, region_annotation_mask

    def step(self, batch, batch_idx, stage, outputs):
        x, y_seq, y_mask, region_mask = self.get_xy(batch)

        # Get clinical features, if used
        clinical_features = self.get_clinical_features(batch)

        # Get risk scores and activation maps from your model
        y_hat, activation_map = self.forward(x, return_maps=True, added_features=clinical_features)  # y_hat: (B, T), activation_map: (B, C, D, H, W)
        
        # Reduce activation map to single channel; normalize using sigmoid
        activation_map_reduced, activation_map_normalized = self.get_activation_map_reduced_normalized(activation_map)
        
        # Resize region mask to match activation map size
        region_mask_resized = self.get_region_mask_resized(region_mask, activation_map)

        # Mask for right-censored follow-up in patients without cancer
        mask = torch.logical_or(torch.cumsum(y_seq, dim=1) > 0, y_mask)  # Corrected dim from 0 to 1

        # calculate losses
        use_localization = False
        classification_loss = self.get_classification_loss(y_hat, y_seq, mask)
        localization_loss = self.get_localization_loss(activation_map_reduced, region_mask_resized) if use_localization else 0
        total_loss = classification_loss + 0.5 * localization_loss  # Adjust weight as needed
        
        # Log metrics
        metric_dict = {
            'classification_loss': classification_loss,
            'localization_loss': localization_loss,
            'loss': total_loss  # must reflect monitor_key in EarlyStopping callback
        }
        
        # Compute accuracy for each year
        for year in range(self.max_followup):
            if mask[:, year].sum() > 0:  # If any valid samples in year
                year_acc = self.accuracy(
                    y_hat[:, year][mask[:, year]],
                    y_seq[:, year][mask[:, year]]
                )
            else:
                year_acc = float('nan')
            metric_dict.update({f'{year+1}year_acc': year_acc})
        
        # Log metrics to wandb
        for metric_name, metric_value in metric_dict.items():
            self.log(f'{stage}_{metric_name}', metric_value, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)
        
        # Store the predictions and labels for use at the end of the epoch for AUC and C-Index computation
        outputs.append({
            "y_hat": y_hat.detach(),  # Logits for all risk scores
            "y_mask": y_mask,  # Tensor of when the patient was observed
            "y_seq": y_seq,  # Tensor of when the patient had cancer
            "y": batch["y"],  # If patient has cancer within self.max_followup years (bool)
            "time_at_event": batch["time_at_event"],  # Censor time (int)
            "attention_map": activation_map_normalized.detach(),
            "true_masks": region_mask_resized,
            "criteria": batch[self.trainer.datamodule.criteria],
            **{k: batch[k] for k in self.trainer.datamodule.group_keys}
        })

        return total_loss

    def get_clinical_features(self, batch):
        if self.clinical_features:
            clinical_features_dict = {}
            for k in self.clinical_features:
                # get feature type
                feature_type = clinical_feature_type[k]

                # vectorize clinical feature as numpy array
                clinical_feature_k = self.trainer.datamodule.vectorizer.transform(
                    {k: self.safely_to_numpy(batch[k])}, feature_type=feature_type
                )

                if feature_type == 'categorical':
                    clinical_feature_k[k] = np.argmax(clinical_feature_k[k], axis=1, keepdims=True)

                # send back to torch
                clinical_features_dict.update({k:torch.from_numpy(v.squeeze(1)) for k,v in clinical_feature_k.items()})

            # concatenate all features
            clinical_features = torch.stack(list(clinical_features_dict.values())).cuda().bfloat16().T # size: (B, len(clinical_features))
        else:
            clinical_features = None
        
        return clinical_features
    
    def get_classification_loss(self, y_hat, y_seq, mask):
        classification_loss = F.binary_cross_entropy_with_logits(y_hat, y_seq, reduction='none')
        classification_loss = (classification_loss * mask).sum() / mask.sum()  # Masked average
        return classification_loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train", self.training_outputs)
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val", self.validation_outputs)
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test", self.test_outputs)

    def on_epoch_end(self, stage, outputs):
        y_hat = F.sigmoid(torch.cat([o["y_hat"] for o in outputs])) # get probabilities from logits
        y_seq = torch.cat([o["y_seq"] for o in outputs])
        y_mask = torch.cat([o["y_mask"] for o in outputs])

        # calculate metrics by year
        for i in range(self.max_followup):
            '''
                Filter samples for either valid negative (observed followup) at time i
                or known pos within range i (including if cancer at prev time and censoring before current time)
            '''
            valid_probs = y_hat[:, i][(y_mask[:, i] == 1) | (y_seq[:,i] == 1)]
            valid_labels = y_seq[:, i][(y_mask[:, i] == 1)| (y_seq[:,i] == 1)]

            # auc
            self.log("{}_{}year_auc".format(stage, i+1), self.auc(valid_probs, valid_labels.view(-1)), sync_dist=True, prog_bar=True)

            # confusion matrix metrics
            confmat_metrics = self.get_confmat_metrics(valid_probs, valid_labels.int())
            for metric_name, metric_value in confmat_metrics.items():
                self.log(f"{stage}_{i+1}year_{metric_name}", metric_value, sync_dist=True, prog_bar=True)

        # calculate concordance index
        y = torch.cat([o["y"] for o in outputs])
        time_at_event = torch.cat([o["time_at_event"] for o in outputs])

        if y.sum() > 0 and self.max_followup == 6:
            c_index = concordance_index(time_at_event.cpu().numpy(), y_hat.detach().cpu().numpy(), y.cpu().numpy(), NLST_CENSORING_DIST)
        else:
            c_index = 0
        self.log("{}_c_index".format(stage), c_index, sync_dist=True, prog_bar=True)

        # Compute localization metrics
        attention_maps = torch.cat([o["attention_map"] for o in outputs])
        true_masks = torch.cat([o["true_masks"] for o in outputs])

        iou_score = self.iou_metric(attention_maps, true_masks.int())
        dice_score = self.dice_metric(attention_maps, true_masks.int())

        self.log(f'{stage}_IoU', iou_score, sync_dist=True, prog_bar=True)
        self.log(f'{stage}_Dice', dice_score, sync_dist=True, prog_bar=True)

    def on_train_epoch_end(self):
        self.on_epoch_end("train", self.training_outputs)
        self.training_outputs = []

    def on_validation_epoch_end(self):
        self.on_epoch_end("val", self.validation_outputs)
        self.roc_analysis_across_nodes(self.validation_outputs, 'val set')
        self.validation_outputs = []

    def on_test_epoch_end(self):
        self.on_epoch_end("test", self.test_outputs)
        self.roc_analysis_across_nodes(self.test_outputs, 'test set')
        self.test_outputs = []

    def get_outputs_across_samples(self, split_outputs, output_keys=['y_seq', 'y_hat', 'y_mask', 'criteria']):
        # collect outputs across samples into a single tensor
        output_across_samples = {}
        for k in [*output_keys, *self.trainer.datamodule.group_keys]: 
            output = torch.cat([o[k] for o in split_outputs])
            if k == 'y_hat':
                output = F.sigmoid(output)
            output_across_samples.update({k: output})
        
        return output_across_samples

    def roc_analysis(self, output_across_nodes, plot_label):
        # unpack data
        y_seq = np.where(output_across_nodes['y_mask'], output_across_nodes['y_seq'], 0)
        y_hat = np.where(output_across_nodes['y_mask'], output_across_nodes['y_hat'], 0)
        criteria = np.where(output_across_nodes['y_mask'], np.tile(output_across_nodes['criteria'], (self.max_followup, 1)).T, 0)
        group_data = {k:v for k,v in output_across_nodes.items() if k in self.trainer.datamodule.group_keys}

        # ensure 2D for plotting
        y_hat = y_hat[:, np.newaxis] if y_hat.ndim == 1 else y_hat
        y_seq = y_seq[:, np.newaxis] if y_seq.ndim == 1 else y_seq

        # generate plots
        roc_plot = self.generate_followup_roc(y_seq, y_hat, criteria)
        print('ROC curve generated.')

        subgroup_roc_plots_by_year = []
        if group_data:
            for year in range(self.max_followup):
                subgroup_roc_plot = self.generate_subgroup_roc(y_seq[:, year], 
                                                               y_hat[:, year], 
                                                               criteria[:, year],
                                                               group_data,
                                                               plot_label=f'{self.trainer.datamodule.criteria}, year{year+1}')
                subgroup_roc_plots_by_year.append(subgroup_roc_plot)
                print(f'Subgroup ROC curve generated for year {year+1}.')
        
        # generate plot names
        plot_name = f'ROC, {plot_label}, full {self.max_followup}-year followup, epoch{self.current_epoch}'
        subgroup_plot_names_by_year = [f'ROC by subgroups, {plot_label} @year{year+1}, epoch{self.current_epoch}' 
                                       for year in range(self.max_followup)]
        
        # log plots
        if self.logger.experiment:
            wandb_logger = self.logger
            wandb_logger.log_image(key=plot_name, images=[roc_plot])
            if group_data:
                for year in range(self.max_followup):
                    wandb_logger.log_image(key=subgroup_plot_names_by_year[year], images=[subgroup_roc_plots_by_year[year]])


    def generate_followup_roc(self, y, y_hat, criteria):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
        colors = cm.rainbow(np.linspace(0, 1, self.max_followup))

        for year in range(self.max_followup):
            plot_chance_level = True if year == self.max_followup - 1 else False # plot on final year

            self.plot_roc_operation_point(y[:, year],
                                        criteria[:, year],
                                        ax=ax,
                                        plot_label=f'{self.trainer.datamodule.criteria}, year{year+1}',
                                        color=colors[year])

            RocCurveDisplay.from_predictions(y[:, year],
                                            y_hat[:, year],
                                            name=f'year{year+1}',
                                            plot_chance_level=plot_chance_level,
                                            ax=ax,
                                            color=colors[year])

        ax.legend(loc='lower right', prop={'size': 8})
        plt.close()

        return fig
