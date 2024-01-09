import sys
import os
sys.path.append(os.getcwd())

import torch
import numpy as np
from scripts.kilter_utils import plot_matrix


def test_model(model, trained_name, angle, difficulty, denormalize=True):    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(trained_name))
    model.eval()

    input_features = torch.tensor([angle, difficulty], dtype=torch.float32).reshape((1, 2)).to(device)

    with torch.no_grad():
        print(input_features.shape)
        output = model(input_features)
        output = output.view(157, 161)
        output = output.cpu().numpy()

    if denormalize:
        output = denormalize_matrix(output)

    return output

def denormalize_matrix(matrix):
    matrix = matrix * 15  # Dénormaisation à [0, 15]
    matrix = np.round(matrix)  # Arrondir à l'entier le plus proche
    return matrix

if __name__ == '__main__':
    from src.neural_network.gan import Generator
    from scripts.kilter_utils import get_all_holes_12x12, get_matrix_from_holes

    model = Generator()
    trained_name = 'models/gan_generator.pt'

    angle = 40
    difficulty = 15

    generated_hold_matrix = test_model(model, trained_name, angle, difficulty)

    num_holds = generated_hold_matrix[generated_hold_matrix > 1].shape
    
    all_holes = get_all_holes_12x12()

    all_holes = get_matrix_from_holes(all_holes)


    print(f'Number of holds: {num_holds}')
    plot_matrix(generated_hold_matrix)




