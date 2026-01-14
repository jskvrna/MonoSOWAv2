import torch
import smplx
import numpy as np
import os

from anno_V3 import AutoLabel3D

class SMPLXLayer(AutoLabel3D):
    def __init__(self, args):
        """
        Initializes the SMPLXLayer.
        Assumes necessary configuration parameters (model_path, device, batch_size, etc.)
        are already set as instance attributes (e.g., by a base class or config loader).
        """
        super().__init__(args)

        self.smplx_model_path: str = self.cfg.paths.smplx_models_path
        self.smplx_gender: str = 'neutral'
        self.smplx_device: str = self.cfg.general.device
        if self.smplx_device == 'gpu':
            self.smplx_device = 'cuda'
        self.smplx_batch_size: int = 1
        self.smplx_num_betas: int = 10
        self.smplx_num_expression_coeffs: int = 10
        self.smplx_ext: str = 'npz'
        self.smplx_use_pca: bool = False
        self.smplx_use_face_contour: bool = True

        # --- Load SMPL-X model once ---
        self.smplx_model = smplx.create(
            model_path=self.smplx_model_path,
            model_type='smplx', # Explicitly set model type
            gender=self.smplx_gender,
            use_face_contour=self.smplx_use_face_contour,
            num_betas=self.smplx_num_betas,
            num_expression_coeffs=self.smplx_num_expression_coeffs,
            ext=self.smplx_ext,
            use_pca=self.smplx_use_pca,
            batch_size=self.smplx_batch_size
        ).to(self.smplx_device)
        self.smplx_faces = self.smplx_model.faces_tensor # Faces are constant

    def generate_smplx_mesh(self,
        betas: torch.Tensor = None,
        body_pose: torch.Tensor = None,
        global_orient: torch.Tensor = None,
        transl: torch.Tensor = None,
        left_hand_pose: torch.Tensor = None,
        right_hand_pose: torch.Tensor = None,
        jaw_pose: torch.Tensor = None,
        leye_pose: torch.Tensor = None,
        reye_pose: torch.Tensor = None,
        expression: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generates an SMPL-X mesh given parameters using the pre-loaded model.
        Note: Input tensor batch size must match the batch_size used during initialization.

        Args:
            betas (torch.Tensor, optional): Shape parameters. Shape: (batch_size, num_betas). Defaults to zeros.
            body_pose (torch.Tensor, optional): Pose parameters for the body joints (excluding root).
                                            Shape: (batch_size, 63) (21 joints * 3 axis-angle). Defaults to zeros.
            global_orient (torch.Tensor, optional): Root orientation. Shape: (batch_size, 3). Defaults to zeros.
            transl (torch.Tensor, optional): Global translation. Shape: (batch_size, 3). Defaults to zeros.
            left_hand_pose (torch.Tensor, optional): Pose parameters for the left hand.
                                                 Shape: (batch_size, 45 or num_pca_comps). Defaults to zeros.
            right_hand_pose (torch.Tensor, optional): Pose parameters for the right hand.
                                                  Shape: (batch_size, 45 or num_pca_comps). Defaults to zeros.
            jaw_pose (torch.Tensor, optional): Pose parameters for the jaw. Shape: (batch_size, 3). Defaults to zeros.
            leye_pose (torch.Tensor, optional): Pose parameters for the left eye. Shape: (batch_size, 3). Defaults to zeros.
            reye_pose (torch.Tensor, optional): Pose parameters for the right eye. Shape: (batch_size, 3). Defaults to zeros.
            expression (torch.Tensor, optional): Expression parameters. Shape: (batch_size, num_expression_coeffs). Defaults to zeros.


        Returns:
            torch.Tensor: Mesh vertices. Shape: (batch_size, 10475, 3).
                          Faces can be accessed via `self.faces`.
        """
        # Determine current batch size from inputs, default to initialized batch size
        input_tensors = [betas, body_pose, global_orient, transl, left_hand_pose,
                 right_hand_pose, jaw_pose, leye_pose, reye_pose, expression]
        current_bs = self.smplx_batch_size
        for tensor in input_tensors:
            if tensor is not None:
                current_bs = tensor.shape[0]
            break

        # Validate batch size
        if current_bs != self.smplx_batch_size:
             # Option 1: Raise Error (strict)
             raise ValueError(f"Input tensor batch size ({current_bs}) must match the model's initialized batch size ({self.smplx_batch_size})")
             # Option 2: Log a warning (flexible, but might hide issues)
             # print(f"Warning: Input tensor batch size ({current_bs}) differs from initialized batch size ({self.batch_size}). Using {current_bs}.")
             # Note: If flexible, the default tensor creation below needs adjustment.
             # For now, sticking to the strict approach based on previous code.

        # --- Create default tensors if parameters are not provided ---
        # Use instance attributes for device, batch_size, num_betas, etc.
        if betas is None:
            betas = torch.zeros((self.smplx_batch_size, self.smplx_num_betas), device=self.smplx_device, dtype=torch.float32)
        if body_pose is None:
            # 21 joints * 3 DOF = 63
            body_pose = torch.zeros((self.smplx_batch_size, 63), device=self.smplx_device, dtype=torch.float32)
        if global_orient is None:
            global_orient = torch.zeros((self.smplx_batch_size, 3), device=self.smplx_device, dtype=torch.float32)
        if transl is None:
            transl = torch.zeros((self.smplx_batch_size, 3), device=self.smplx_device, dtype=torch.float32)

        # Hand poses depend on whether PCA is used (check smplx model config if needed)
        # Assuming full pose (15 joints * 3 DOF = 45) if use_pca is False
        hand_pose_dim = 45 # Adjust if self.use_pca is True and you know the PCA dim
        if left_hand_pose is None:
            left_hand_pose = torch.zeros((self.smplx_batch_size, hand_pose_dim), device=self.smplx_device, dtype=torch.float32)
        if right_hand_pose is None:
            right_hand_pose = torch.zeros((self.smplx_batch_size, hand_pose_dim), device=self.smplx_device, dtype=torch.float32)

        if jaw_pose is None:
            jaw_pose = torch.zeros((self.smplx_batch_size, 3), device=self.smplx_device, dtype=torch.float32)
        if leye_pose is None:
            leye_pose = torch.zeros((self.smplx_batch_size, 3), device=self.smplx_device, dtype=torch.float32)
        if reye_pose is None:
            reye_pose = torch.zeros((self.smplx_batch_size, 3), device=self.smplx_device, dtype=torch.float32)
        if expression is None:
            expression = torch.zeros((self.smplx_batch_size, self.smplx_num_expression_coeffs), device=self.smplx_device, dtype=torch.float32)


        # --- Generate mesh using the pre-loaded model ---
        # Ensure requires_grad is False if not optimizing
        output = self.smplx_model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            expression=expression,
            return_verts=True,
            return_full_pose=False # Usually not needed if just getting vertices
        )

        vertices = output.vertices
        # Faces are accessed via self.faces

        return vertices # Only return vertices, faces are stored in self.faces
    
    def generate_mesh_from_pose_vector(self,
        pose_vector: torch.Tensor,
        betas: torch.Tensor = None,
        transl: torch.Tensor = None,
        expression: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generates an SMPL-X mesh given the flattened pose parameters (165 values)
        and optional shape, translation, and expression parameters.

        Args:
            pose_vector (torch.Tensor): The flattened pose parameters for the SMPL-X model.
                                      Shape: (batch_size, 165). This includes global orientation,
                                      body pose, jaw pose, eye poses, and hand poses concatenated.
            betas (torch.Tensor, optional): Shape parameters. Shape: (batch_size, num_betas). Defaults to zeros.
            transl (torch.Tensor, optional): Global translation. Shape: (batch_size, 3). Defaults to zeros.
            expression (torch.Tensor, optional): Expression parameters. Shape: (batch_size, num_expression_coeffs). Defaults to zeros.

        Returns:
            torch.Tensor: Mesh vertices. Shape: (batch_size, 10475, 3).
                          Faces can be accessed via `self.faces`.

        Raises:
            ValueError: If the input pose_vector tensor does not have the expected
                        dimension (165) or if its batch size doesn't match other inputs
                        or the initialized model batch size (if strict checking is enabled
                        in generate_smplx_mesh).
        """
        if pose_vector.shape[-1] != 165:
            raise ValueError(f"Expected pose_vector to have 165 parameters, but got {pose_vector.shape[-1]}")

        current_bs = pose_vector.shape[0]

        # --- Parse the pose_vector tensor ---
        # Ensure slicing respects the batch dimension
        global_orient = pose_vector[:, 0:3]
        body_pose = pose_vector[:, 3:66]      # 3 + 63 = 66
        jaw_pose = pose_vector[:, 66:69]     # 66 + 3 = 69
        leye_pose = pose_vector[:, 69:72]     # 69 + 3 = 72
        reye_pose = pose_vector[:, 72:75]     # 72 + 3 = 75
        left_hand_pose = pose_vector[:, 75:120]  # 75 + 45 = 120
        right_hand_pose = pose_vector[:, 120:165] # 120 + 45 = 165

        # --- Call the main generation function ---
        vertices = self.generate_smplx_mesh(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            expression=expression
        )

        return vertices

    def save_mesh_obj(self, vertices: torch.Tensor, file_path: str):
        """Saves the first mesh in a batch to an OBJ file using stored faces."""
        if not hasattr(self, 'faces') or self.smplx_faces is None:
            print("Error: Faces not available in SMPLXLayer instance.")
            return
        if vertices.shape[0] == 0:
            print("Error: No vertices provided to save.")
            return

        # Use self.faces
        faces_np = self.smplx_faces.detach().cpu().numpy()
        # Save the first mesh in the batch
        verts_np = vertices[0].detach().cpu().numpy()

        os.makedirs(os.path.dirname(file_path), exist_ok=True) # Ensure directory exists

        with open(file_path, 'w') as f:
            # Write vertices
            for v in verts_np:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write faces (adjusting indices to be 1-based for OBJ)
            for face in faces_np:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        print(f"Mesh saved to {file_path}")
