import os
import math
import yaml
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt

from .network import get_network
from .encoder import get_encoder
from src.render import run_network
from .dataset import TIGREDataset as Dataset

from src.render.ct_geometry_projector import ConeBeam3DProjector
from odl.tomo.util.utility import axis_rotation, rotation_matrix_from_to

def rotation_matrix_to_axis_angle(m):
    angle = np.arccos((m[0,0] + m[1,1] + m[2,2] - 1)/2)

    x = (m[2,1] - m[1,2])/math.sqrt((m[2,1]-m[1,2])**2 + (m[0,2] - m[2,0])**2 + (m[1,0] -m[0,1])**2)
    y = (m[0,2] - m[2,0])/math.sqrt((m[2,1]-m[1,2])**2 + (m[0,2]-m[2,0])**2 + (m[1,0]-m[0,1])**2)
    z = (m[1,0] - m[0,1])/math.sqrt((m[2,1]-m[1,2])**2 + (m[0,2]-m[2,0])**2 + (m[1,0]-m[0,1])**2)
    axis=(x,y,z)

    return axis, angle

class Trainer:
    def __init__(self, cfg, device="cuda"):

        # Args
        self.conf = cfg
        self.epochs = cfg["train"]["epoch"]
        self.netchunk = cfg["render"]["netchunk"]
        
        # Memory optimization settings
        self.use_mixed_precision = cfg.get("train", {}).get("mixed_precision", True)
        self.memory_efficient_eval = cfg.get("train", {}).get("memory_efficient_eval", True)
        
        # Initialize AMP scaler for mixed precision training
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_mixed_precision)
        
        # Setup output directories for batch processing
        base_output_dir = cfg["exp"].get("output_recon_dir", "./logs/reconstructions/")
        self.current_model_id = cfg["exp"].get("current_model_id", 1)
        
        # Create nested folder structure: base_dir/model_id/experiment_name/
        original_model_id = str(self.current_model_id).split('_')[0]
        experiment_name = str(self.current_model_id)
        self.output_recon_dir = osp.join(base_output_dir, original_model_id, experiment_name)
        
        os.makedirs(self.output_recon_dir, exist_ok=True)

        # Load CT geometry configuration
        configPath = cfg['exp']['dataconfig']
        with open(configPath, "r") as handle:
            data = yaml.safe_load(handle)

        # Setup data paths from main config (CCTA.yaml) - much cleaner!
        input_data_dir = cfg["exp"].get("input_data_dir", "./data/GT_volumes/")
        
        # Extract original model ID from experiment name (format: {model_id}_lr{lr}_loss{type})
        original_model_id = str(self.current_model_id).split('_')[0]
        gt_volume_filename = f"{original_model_id}.npy"
        gt_volume_path = osp.join(input_data_dir, gt_volume_filename)
        
        print(f"Processing experiment {self.current_model_id}")
        print(f"Original model ID: {original_model_id}")
        print(f"GT volume path: {gt_volume_path}")

        # Check if ground truth volume exists
        if not os.path.exists(gt_volume_path):
            raise FileNotFoundError(f"Ground truth volume not found: {gt_volume_path}")

        dsd = data["DSD"] # Distance Source Detector   mm   
        dso = data["DSO"] # Distance Source Origin      mm 
        dde = data["DDE"]

        # Detector parameters
        proj_size = np.array(data["nDetector"])  # number of pixels              (px)
        proj_reso = np.array(data["dDetector"]) 

        # Image parameters
        image_size = np.array(data["nVoxel"])  # number of voxels              (vx)
        image_reso = np.array(data["dVoxel"])  # size of each voxel            (mm)
   
        first_proj_angle = [-data["first_projection_angle"][1], data["first_projection_angle"][0]]
        second_proj_angle = [-data["second_projection_angle"][1], data["second_projection_angle"][0]]

        # First_projection
        from_source_vec= (0,-dso[0],0)
        from_rot_vec = (-1,0,0)
        to_source_vec = axis_rotation((0,0,1), angle=first_proj_angle[0]/180*np.pi, vectors=from_source_vec)
        to_rot_vec = axis_rotation((0,0,1), angle=first_proj_angle[0]/180*np.pi, vectors=from_rot_vec)
        to_source_vec = axis_rotation(to_rot_vec[0], angle=first_proj_angle[1]/180*np.pi, vectors=to_source_vec[0])

        rot_mat = rotation_matrix_from_to(from_source_vec, to_source_vec[0])
        proj_axis, proj_angle = rotation_matrix_to_axis_angle(rot_mat)

        self.ct_projector_first = ConeBeam3DProjector(image_size, image_reso, proj_angle, proj_axis, proj_size, proj_reso, dde[0], dso[0])

        # Second_projection
        from_source_vec= (0,-dso[1],0)
        from_rot_vec = (-1,0,0)
        to_source_vec = axis_rotation((0,0,1), angle=second_proj_angle[0]/180*np.pi, vectors=from_source_vec)
        to_rot_vec = axis_rotation((0,0,1), angle=second_proj_angle[0]/180*np.pi, vectors=from_rot_vec)
        to_source_vec = axis_rotation(to_rot_vec[0], angle=second_proj_angle[1]/180*np.pi, vectors=to_source_vec[0])

        rot_mat = rotation_matrix_from_to(from_source_vec, to_source_vec[0])
        proj_axis, proj_angle = rotation_matrix_to_axis_angle(rot_mat)

        self.ct_projector_second = ConeBeam3DProjector(image_size, image_reso, proj_angle, proj_axis, proj_size, proj_reso, dde[1], dso[1])
        
        # Load 3D ground truth volume and generate projections (simplified)
        phantom = np.load(gt_volume_path)
        phantom = np.transpose(phantom, (1,2,0))[::,::-1,::-1]
        phantom = np.transpose(phantom, (2,1,0))[::-1,::,::].copy()
        phantom = torch.tensor(phantom, dtype=torch.float32)[None, ...]

        train_projs_one = self.ct_projector_first.forward_project(phantom)
        train_projs_two = self.ct_projector_second.forward_project(phantom)

        data["projections"] = torch.cat((train_projs_one,train_projs_two), 1)
        print(f"Generated projections from 3D volume: {data['projections'].shape}")
        
        # Dataset preparation based on mode
        self.use_sdf = cfg.get("train", {}).get("use_sdf", True)
        
        # Always generate 2D SDF targets from ground truth occupancy projections
        # (needed for SDF loss computation regardless of use_sdf mode)
        from src.render.sdf_utils import occupancy_to_sdf_2d
        proj_sdf_one = occupancy_to_sdf_2d(train_projs_one.squeeze(0).squeeze(0), voxel_size=proj_reso[0])  # [512, 512]
        proj_sdf_two = occupancy_to_sdf_2d(train_projs_two.squeeze(0).squeeze(0), voxel_size=proj_reso[1])  # [512, 512]
        data["sdf_projections"] = torch.cat((proj_sdf_one[None, None, :], proj_sdf_two[None, None, :]), 1)  # [1, 2, 512, 512]
        print(f"Generated 2D SDF targets from GT occupancy projections: {data['sdf_projections'].shape}")

        # Dataset
        self.dataconfig = data
        self.train_dset = Dataset(data, device)
        self.voxels = self.train_dset.voxels
        
        # Set last_activation based on use_sdf
        cfg["network"]["use_sdf"] = True if self.use_sdf else False
            
        # Load loss weights from config (respect user's settings regardless of use_sdf)
        self.loss_weights = cfg.get("train", {}).get("current_loss_weights", [1.0, 1.0])
        self.projection_weight, self.sdf_loss_weight = self.loss_weights
        
        print(f"SDF Mode: {self.use_sdf}")
        print(f"Loss Weights - Projection: {self.projection_weight}, SDF: {self.sdf_loss_weight}")

        # Network
        net_type = cfg["network"]["net_type"]
        network = get_network(net_type)
        encoder = get_encoder(**cfg["encoder"])
        network_config = {k: v for k, v in cfg["network"].items() if k != "net_type"}
        self.net = network(encoder, **network_config).to(device)
        self.grad_vars = list(self.net.parameters())

        # Optimizer with memory-efficient settings
        weight_decay_val = cfg["train"].get("weight_decay", 1e-6)
        if isinstance(weight_decay_val, str):
            weight_decay_val = float(weight_decay_val)
            
        self.optimizer = torch.optim.AdamW(
            params=self.grad_vars, 
            lr=cfg["train"]["lrate"], 
            betas=(0.9, 0.999),
            weight_decay=weight_decay_val,  # Ensure proper type conversion
            eps=1e-8
        )

        self.training_losses = []
        
        # Best model tracking
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_model_state = None

    def save_loss_plot(self):
        """
        Save training loss plot for current model.
        """
        if len(self.training_losses) == 0:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_losses, 'b-', linewidth=1.5)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title(f'Training Loss - Model {self.current_model_id}')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization
        
        loss_plot_path = osp.join(self.output_recon_dir, f"loss_plot_{self.current_model_id}.png")
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved loss plot: {loss_plot_path}")

    def start(self):
        """
        Main loop with memory optimizations.
        """
        for idx_epoch in tqdm(range(1, self.epochs+1)):
            
            # Clear cache before evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Train
            self.net.train()
            
            # Memory-efficient training step
            loss_train = self.train_step_memory_efficient(self.train_dset)
            
            # Track loss for plotting
            current_loss = loss_train['loss']
            self.training_losses.append(current_loss)
            
            print(f"epoch={idx_epoch}/{self.epochs}, loss={current_loss:.6f}")
            
        # Save loss plot after training completion
        self.save_loss_plot()
        
        # Evaluate and save the last model at the end
        print(f"\nEvaluating last model from epoch {self.epochs}")
        
        # Evaluate last model (current state)
        self.net.eval()
        with torch.no_grad():
            if self.memory_efficient_eval:
                eval_chunk_size = self.netchunk // 4
                model_pred = self.run_network_chunked(
                    self.voxels, 
                    self.net,
                    eval_chunk_size
                )
            else:
                model_pred = run_network(self.voxels, self.net, self.netchunk)
            
            model_pred = (model_pred.squeeze()).detach().cpu().numpy()
            
            # Save last model results with all the comprehensive outputs
            self._save_best_model_results(model_pred, self.epochs)
        
        tqdm.write(f"Training complete! Saved last model from epoch {self.epochs}")
        
    def _save_best_model_results(self, model_pred, epoch):
        """Save comprehensive results for the best model."""
        print(f"Saving best model results from epoch {epoch}...")
        
        # Handle SDF vs Occupancy modes differently
        pred_tensor = torch.tensor(model_pred, dtype=torch.float32, device=self.train_dset.projs.device)[None, ...]
        
        if self.use_sdf:
            # SDF mode: model generates SDF, we convert to occupancy for projections
            print("SDF mode: Model generated 3D SDF")
            sdf_3d_filename = f"sdf_3d_{self.current_model_id}.npy"
            sdf_3d_path = osp.join(self.output_recon_dir, sdf_3d_filename)
            np.save(sdf_3d_path, model_pred)
            print(f"Saved 3D SDF prediction: {sdf_3d_path}")
            
            # Convert SDF to occupancy for projections
            from src.render.sdf_utils import sdf_to_occupancy
            occupancy_for_proj = sdf_to_occupancy(pred_tensor, alpha=50.0)
            
            # Save converted occupancy
            occupancy_3d_filename = f"recon_occupancy_{self.current_model_id}.npy"
            occupancy_3d_path = osp.join(self.output_recon_dir, occupancy_3d_filename)
            occupancy_3d_data = occupancy_for_proj.squeeze().detach().cpu().numpy()
            np.save(occupancy_3d_path, occupancy_3d_data)
            print(f"Saved occupancy converted from SDF: {occupancy_3d_path}")
            
        else:
            # Occupancy mode: model generates occupancy directly
            print("Occupancy mode: Model generated 3D occupancy")
            occupancy_3d_filename = f"recon_occupancy_{self.current_model_id}.npy"
            occupancy_3d_path = osp.join(self.output_recon_dir, occupancy_3d_filename)
            np.save(occupancy_3d_path, model_pred)
            print(f"Saved 3D occupancy prediction: {occupancy_3d_path}")
            
            occupancy_for_proj = pred_tensor
        
        # Save all other comprehensive outputs (GT, projections, images, comparisons, network)
        self._save_comprehensive_outputs(pred_tensor, occupancy_for_proj, epoch)

    def run_network_chunked(self, inputs, fn, chunk_size):
        """
        Memory-efficient network inference with smaller chunks.
        """
        uvt_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        output_chunks = []
        
        for i in range(0, uvt_flat.shape[0], chunk_size):
            chunk = uvt_flat[i:i + chunk_size]
            with torch.amp.autocast('cuda', enabled=self.use_mixed_precision, dtype=torch.float16):
                chunk_output = fn(chunk)
            output_chunks.append(chunk_output)
            
            # Clear intermediate tensors
            del chunk, chunk_output
            if torch.cuda.is_available() and i % (chunk_size * 4) == 0:  # Periodic cleanup
                torch.cuda.empty_cache()
        
        out_flat = torch.cat(output_chunks, 0)
        out = out_flat.reshape(list(inputs.shape[:-1]) + [out_flat.shape[-1]])
        
        # Clean up
        del output_chunks, out_flat
        
        return out

    def train_step_memory_efficient(self, data):
        """
        Memory-efficient training step with gradient accumulation and mixed precision.
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        
        # Gradient accumulation loop
        with torch.amp.autocast('cuda', enabled=self.use_mixed_precision, dtype=torch.float16):
            loss = self.compute_loss(data)
            total_loss += loss["loss"].item()
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss["loss"]).backward()
        
        # Clear intermediate tensors
        del loss

        # Optimizer step with gradient scaling
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Clear gradients and cache
        self.optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {"loss": total_loss}

    def train_step(self, data):
        """
        Legacy training step - kept for compatibility
        """
        return self.train_step_memory_efficient(data)
        
    def _save_comprehensive_outputs(self, sdf_pred_tensor, occupancy_pred, epoch):
        """Save comprehensive outputs including projections, images, and comparisons."""
        # Save ground truth 3D model
        gt_volume_filename = f"gt_volume_{self.current_model_id}.npy"
        gt_volume_path = osp.join(self.output_recon_dir, gt_volume_filename)
        input_data_dir = self.conf["exp"].get("input_data_dir", "./data/GT_volumes/")
        original_model_id = str(self.current_model_id).split('_')[0]
        original_gt_path = osp.join(input_data_dir, f"{original_model_id}.npy")
        gt_volume = np.load(original_gt_path)
        np.save(gt_volume_path, gt_volume)
        print(f"Saved ground truth 3D model: {gt_volume_path}")
        
        # Save ground truth projections
        gt_projs_filename = f"gt_projections_{self.current_model_id}.npy"
        gt_projs_path = osp.join(self.output_recon_dir, gt_projs_filename)
        gt_projs_data = self.train_dset.projs.detach().cpu().numpy()
        np.save(gt_projs_path, gt_projs_data)
        print(f"Saved ground truth projections: {gt_projs_path}")
        
        # Generate predicted projections
        pred_projs_one = self.ct_projector_first.forward_project(occupancy_pred)
        pred_projs_two = self.ct_projector_second.forward_project(occupancy_pred)
        pred_projs = torch.cat((pred_projs_one, pred_projs_two), 1)
        
        pred_projs_filename = f"pred_projections_{self.current_model_id}.npy"
        pred_projs_path = osp.join(self.output_recon_dir, pred_projs_filename)
        pred_projs_data = pred_projs.detach().cpu().numpy()
        np.save(pred_projs_path, pred_projs_data)
        print(f"Saved predicted projections: {pred_projs_path}")
        
        # Always save ground truth 2D SDF (generated for all modes now)
        gt_sdf_2d_filename = f"sdf_2d_gt_{self.current_model_id}.npy"
        gt_sdf_2d_path = osp.join(self.output_recon_dir, gt_sdf_2d_filename)
        gt_sdf_2d_data = self.dataconfig["sdf_projections"].detach().cpu().numpy()
        np.save(gt_sdf_2d_path, gt_sdf_2d_data)
        print(f"Saved 2D SDF ground truth: {gt_sdf_2d_path}")
        
        # Generate and save 2D SDF predictions if model was trained with SDF loss
        if self.sdf_loss_weight > 0:
            detector_pixel_size = self.dataconfig["dDetector"][0]
            
            if self.use_sdf:
                # SDF mode: convert 3D SDF to 2D SDF via occupancy
                from src.render.sdf_utils import sdf_3d_to_occupancy_to_sdf_2d
                pred_sdf_2d, _ = sdf_3d_to_occupancy_to_sdf_2d(
                    sdf_pred_tensor, self.ct_projector_first, self.ct_projector_second,
                    alpha=50.0, voxel_size_2d=detector_pixel_size
                )
            else:
                # Occupancy mode: convert 2D occupancy projections to 2D SDF
                from src.render.sdf_utils import occupancy_to_sdf_2d
                sdf_2d_view1 = occupancy_to_sdf_2d(pred_projs[0, 0], voxel_size=detector_pixel_size)
                sdf_2d_view2 = occupancy_to_sdf_2d(pred_projs[0, 1], voxel_size=detector_pixel_size)
                pred_sdf_2d = torch.stack([sdf_2d_view1, sdf_2d_view2], dim=0)[None, ...]
            
            pred_sdf_2d_filename = f"sdf_2d_pred_{self.current_model_id}.npy"
            pred_sdf_2d_path = osp.join(self.output_recon_dir, pred_sdf_2d_filename)
            pred_sdf_2d_data = pred_sdf_2d.detach().cpu().numpy()
            np.save(pred_sdf_2d_path, pred_sdf_2d_data)
            print(f"Saved 2D SDF predictions: {pred_sdf_2d_path}")
        else:
            # No SDF loss used during training
            pred_sdf_2d_data = None
            print("No SDF loss used - skipping 2D SDF prediction generation")
        
        # Create comparison images based on whether SDF loss was used
        comparison_path = osp.join(self.output_recon_dir, f"comparison_{self.current_model_id}.png")
        
        if self.sdf_loss_weight > 0 and pred_sdf_2d_data is not None:
            # SDF mode: show both occupancy and SDF projections
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            for i in range(2):  # Two views
                # Ground truth occupancy projection
                axes[i, 0].imshow(gt_projs_data[0, i], cmap='gray')
                axes[i, 0].set_title(f'GT Occupancy View {i+1}')
                axes[i, 0].axis('off')
                
                # Predicted occupancy projection
                axes[i, 1].imshow(pred_projs_data[0, i], cmap='gray')
                axes[i, 1].set_title(f'Pred Occupancy View {i+1}')
                axes[i, 1].axis('off')
                
                # Ground truth SDF (if available)
                if gt_sdf_2d_data is not None:
                    axes[i, 2].imshow(gt_sdf_2d_data[0, i], cmap='RdBu_r')
                    axes[i, 2].set_title(f'GT SDF View {i+1}')
                    axes[i, 2].axis('off')
                else:
                    axes[i, 2].text(0.5, 0.5, 'No GT SDF', ha='center', va='center')
                    axes[i, 2].axis('off')
                
                # Predicted SDF
                axes[i, 3].imshow(pred_sdf_2d_data[0, i], cmap='RdBu_r')
                axes[i, 3].set_title(f'Pred SDF View {i+1}')
                axes[i, 3].axis('off')
        else:
            # Occupancy mode: only show occupancy projections
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            
            for i in range(2):  # Two views
                # Ground truth occupancy projection
                axes[i, 0].imshow(gt_projs_data[0, i], cmap='gray')
                axes[i, 0].set_title(f'GT Occupancy View {i+1}')
                axes[i, 0].axis('off')
                
                # Predicted occupancy projection
                axes[i, 1].imshow(pred_projs_data[0, i], cmap='gray')
                axes[i, 1].set_title(f'Pred Occupancy View {i+1}')
                axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison image: {comparison_path}")
        
        # Save network weights
        network_filename = f"network_{self.current_model_id}.pth"
        network_path = osp.join(self.output_recon_dir, network_filename)
        torch.save({
            'network': self.net.state_dict(),
            'model_id': self.current_model_id,
            'epoch': epoch,
            'config': self.conf
        }, network_path)
        print(f"Saved last network: {network_path}")

    def compute_loss(self, data):
        """
        Training step
        """
        raise NotImplementedError()
        