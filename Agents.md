You are to perform a targeted architectural adaptation of the provided canonical PyTorch implementation of Perceiver IO. The goal is to transform the model from a general perception architecture into a specialized **generative optimizer for neural network weights**.

You will start with the provided high-fidelity Perceiver IO code as a baseline. You will then introduce new modules and modify the existing structure to handle the unique input and output format of this task, which is a model's `state_dict`.

### **1. System Architecture**

The final, adapted model must be a modular system composed of the following distinct `torch.nn.Module` classes:

*   **`InputAdapter` (New):** Preprocesses an input `state_dict`.
*   **`OutputAdapter` (New):** Reconstructs the final `state_dict` from the model's output.
*   **`MixtureOfExperts` (New):** An MoE layer to replace the standard MLP.
*   **`LatentProcessorWithMoE` (Modified):** A modified version of the latent self-attention block.
*   **`GenerativePerceiver` (New):** The top-level module that encapsulates and connects all components.
*   **`PerceiverEncoder` and `BasicDecoder` (Existing):** You will reuse the verified `PerceiverEncoder` and `BasicDecoder` modules from the baseline implementation.

### **2. Component Specifications**

**a. `InputAdapter` (New Module)**

*   **Input:** A Python dictionary representing a model's `state_dict`.
*   **Function:** It must iterate through each parameter `(key, tensor)` in the `state_dict`. For each parameter, it will generate a single, unified input vector for the Perceiver by concatenating two feature sets:
    1.  The **flattened numerical data** from the parameter `tensor`.
    2.  A **learned embedding** derived from the parameter's `key` (e.g., using an embedding layer on a vocabulary of keys).
*   **Output:** It must return two items:
    1.  A 2D tensor of shape `(num_parameters, feature_dim)` representing the full input sequence.
    2.  A `metadata` object (e.g., a list of tuples) containing the original key and shape of each parameter, which is required for reconstruction.

**b. `MixtureOfExperts` (New Module)**

*   **Function:** This module implements conditional computation. It must contain:
    1.  A **gating network** (e.g., a simple `nn.Linear` layer) that takes an input tensor and outputs a probability distribution over the experts.
    2.  A set of `N` **expert networks** (e.g., simple `MLP`s).
*   **Process:** For each input token, the gating network selects one or more experts, which then process the token.

**c. `LatentProcessorWithMoE` (Modified `SelfAttentionBlock`)**

*   **Modification:** You will create a new module, `LatentProcessorWithMoE`, which is a copy of the original `SelfAttentionBlock`. In this new module, you will **replace the standard `MLP` with your new `MixtureOfExperts` module.**

**d. `OutputAdapter` (New Module)**

*   **Input:** The 2D output tensor from the `PerceiverDecoder` and the `metadata` object from the `InputAdapter`.
*   **Function:** It must reverse the `InputAdapter`'s process. It will iterate through the output tensor, and for each vector, it will use the corresponding metadata (key and original shape) to reshape the vector back into its correct tensor form.
*   **Output:** A final `state_dict` dictionary with keys and tensor shapes identical to the original input.

**e. `GenerativePerceiver` (New Top-Level Module)**

*   **Function:** This module orchestrates the entire process. Its `forward` method will execute the full, end-to-end data flow:
    1.  Pass the input `state_dict` to the `InputAdapter` to get the input tensor and metadata.
    2.  Use this tensor as the input to the `PerceiverEncoder`.
    3.  Process the resulting latents through a series of `LatentProcessorWithMoE` blocks.
    4.  Use the processed latents and the original input tensor (as the query) for the `BasicDecoder`.
    5.  Pass the decoder's output and the metadata to the `OutputAdapter` to reconstruct the final `state_dict`.

### **3. Critical Implementation Constraints**

*   **Architectural Generalization:** The implementation must **not** be hardcoded to a specific target model's parameter count or tensor shapes. The adapters must be able to process any `state_dict` passed to them.
*   **Preservation of Baseline:** Do not modify the original `PerceiverEncoder` or `BasicDecoder` modules. Treat them as stable, imported components.

### **4. Deliverable Requirements**

You will provide a **single, complete, and executable Python script** that includes the original baseline code plus all the new and modified modules. The script's `if __name__ == "__main__":` block must be updated to perform a new end-to-end test that verifies the correctness of the **adapted** architecture. This test must:

1.  Define a simple, dummy CNN `torch.nn.Module` (e.g., with two convolutional layers and one linear layer).
2.  Instantiate this dummy CNN and get its `state_dict`.
3.  Instantiate the complete `GenerativePerceiver` model.
4.  Perform a full forward pass, feeding the dummy `state_dict` into the `GenerativePerceiver`.
5.  Receive the output `state_dict` from the model.
6.  **Assert** that the keys of the output `state_dict` are identical to the keys of the input `state_dict`.
7.  **Assert** that the shape of each tensor in the output `state_dict` is identical to the shape of the corresponding tensor in the input `state_dict`.

This verification is non-negotiable and will serve as definitive proof that the adaptation was successful and the crucial `InputAdapter` and `OutputAdapter` are functioning correctly.
