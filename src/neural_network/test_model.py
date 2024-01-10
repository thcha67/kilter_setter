import sys
import os
sys.path.append(os.getcwd())

import torch
import numpy as np
from scripts.kilter_utils import plot_matrix, denormalize_data
import matplotlib.pyplot as plt


def test_model(model, trained_name, angle, difficulty):    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(trained_name))
    model.eval()

    input_features = torch.tensor([angle, difficulty], dtype=torch.float32).reshape((1, 2)).to(device)

    with torch.no_grad():
        output = model(input_features)
        output = output.view(40, 37)
        output = output.cpu().numpy()

    return output

if __name__ == '__main__':
    from src.neural_network.gan import Generator

    model = Generator()
    trained_name = 'models/gan_generator.pt'

    angle = 40/70
    difficulty = 15/70

    generated_hold_matrix = test_model(model, trained_name, angle, difficulty)

    plt.imshow(generated_hold_matrix*70)
    plt.show()
    # num_holds = generated_hold_matrix[generated_hold_matrix > 1].shape
    
    # all_holes = get_all_holes_12x12()

    # all_holes = get_matrix_from_holes(all_holes)


    # print(f'Number of holds: {num_holds}')
    # plot_matrix(generated_hold_matrix)




