import os
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class Custom2DBraTSDataset(Dataset):
    def __init__(self, data_dir, modality, n):
        self.data_dir = data_dir
        self.modality = modality
        self.n = n
        self.patient_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.slice_indices = self._generate_slice_indices()

    def _generate_slice_indices(self):
        slice_indices = []
        for patient_id in self.patient_ids:
            patient_path = os.path.join(self.data_dir, patient_id)
            image_path = os.path.join(patient_path, f'{patient_id}_{self.modality}.nii.gz')
            if os.path.exists(image_path):
                image = nib.load(image_path).get_fdata()
                num_slices = image.shape[2]
                slice_indices.extend([(patient_id, slice_idx) for slice_idx in range(num_slices // 2 - self.n, num_slices // 2 + self.n)])
        return slice_indices

    def __getitem__(self, idx):
        patient_id, slice_idx = self.slice_indices[idx]
        patient_path = os.path.join(self.data_dir, patient_id)

        # Load image and label volumes
        image_path = os.path.join(patient_path, f'{patient_id}_{self.modality}.nii.gz')
        label_path = os.path.join(patient_path, f'{patient_id}_seg.nii.gz')

        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # Extract the specific slice
        image_slice = image[:, :, slice_idx]
        label_slice = label[:, :, slice_idx]

        # Convert to torch tensor and add channel dimension for image
        image_tensor = torch.tensor(image_slice, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # Resize image and label to 256x256
        image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        label_slice = torch.tensor(label_slice, dtype=torch.float32).unsqueeze(0)  # Add channel dimension for resizing
        label_tensor = F.interpolate(label_slice.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)

        label_tensor[label_tensor == 4] = 3

        return image_tensor, label_tensor

    def __len__(self):
        return len(self.slice_indices)