import torch
from art.attacks.evasion import SaliencyMapMethod
from tqdm import trange

def generate_adversarial_samples(data, labels, art_model, N=10, theta=0.8, gamma=0.7, batch_size=32, device=None):
    """
    Generate adversarial samples using the Saliency Map Method (JSMA).

    Args:
        data (np.array): Input data (images).
        labels (np.array): True labels for the data.
        art_model: ART model for generating adversarial samples.
        N : Number of classes.
        theta : Parameter for the attack.
        gamma : Parameter for the attack.
        batch_size : Batch size for generating adversarial samples.

    Returns:
        expanded_data (torch.Tensor): Repeated original data to match adversarial samples.
        expanded_labels (torch.Tensor): Repeated original labels.
        adversarial_data (torch.Tensor): Generated adversarial samples.
        adversarial_targets (torch.Tensor): Target labels for the adversarial samples.
    """


    if device is None:
        device = "cpu"
    elif device=='mps':
        art_model.model.to('cpu')


    adversarial_data = []
    expanded_data = []
    expanded_labels = []
    adversarial_targets = []

    # Create the attack method
    attack = SaliencyMapMethod(classifier=art_model, theta=theta, gamma=gamma, batch_size=batch_size, verbose=False)

    for i in trange(len(data), desc='JSMA'):
        original_sample = data[i:i+1]  # Keep as batch for ART compatibility
        original_label = labels[i]

        for target_class in range(N):
            if target_class != original_label:
                # Generate adversarial sample targeting this class
                adv_sample = attack.generate(x=original_sample, y=torch.tensor([target_class]).unsqueeze(0))
                adv_sample_tensor = torch.from_numpy(adv_sample).squeeze(0)
                original_sample_tensor = original_sample.squeeze(0)
                if isinstance(original_sample_tensor, torch.Tensor) is False:
                    original_sample_tensor = torch.from_numpy(original_sample_tensor)
                adversarial_data.append(adv_sample_tensor)
                expanded_data.append(original_sample_tensor)
                expanded_labels.append(original_label)
                adversarial_targets.append(target_class)

    # Convert lists to tensors
    adversarial_data = torch.stack(adversarial_data)
    expanded_data = torch.stack(expanded_data)
    expanded_labels = torch.tensor(expanded_labels)
    adversarial_targets = torch.tensor(adversarial_targets)

    # Return model to device if mps
    art_model.model.to(device)

    return expanded_data, expanded_labels, adversarial_data, adversarial_targets
