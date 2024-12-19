import torch


def create_true_loadings(
    num_factors, num_variables, block_size, overlap, random=False, mean=1, std=5
):
    B = torch.zeros((num_variables, num_factors))
    block_gap = block_size - overlap

    start_rows = 0
    col = 0

    while start_rows + block_size <= num_variables and col < num_factors:
        end_row = min(start_rows + block_size, num_variables)
        if random:
            B[start_rows:end_row, col] = torch.rand(
                loc=mean, scale=std, size=(end_row - start_rows)
            )
        else:
            B[start_rows:end_row, col] = 1

        start_rows += block_gap
        col += 1

    return B
