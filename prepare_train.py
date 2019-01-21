import numpy as np
from input_funcs import get_filenames, prepare_input

x, y = prepare_input('./data/', get_filenames())
np.save('./temp/x', x)
np.save('./temp/y', y)