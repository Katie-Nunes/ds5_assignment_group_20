# This is some psuedocode I wrote for this exercise, it is very very rough,
# I will make a clearer outline for this later

# draw_mandel(width)

	# Calculate_Mandel
		# define_mandel
			# x_range = [-1.5, 0.5]
			# y_range =  [-1, 1]
		# define_div_index
		# calculate_div_index_per
	# Visualize_Mandel
		# Create Plot in right size
		# Populate plot with pixels depending on div_index

import numpy as np
def map_pixel_to_complex(i: int,j: int, width: int, x_range: tuple(float,float) = (-1.5,0.5),y_range: tuple(float,float) = (-1,1)) -> complex:
    ## Maps a pixel coordinate (i,j) to a point in the complex plane
    
	x_min,x_max = x_range
	y_min,y_max = y_range
	x = x_min + (j / width) * (x_max - x_min)
	y = y_min + (i / width) * (y_max - y_min)
	return complex(x,y)

def mandelbrot_iteration(c: complex, max_iter: int = 100) -> int:
    """
    Calculate how many iterations until the sequence diverges.
    
    Args:
        c: Complex number to test
        max_iter: Maximum iterations (100)
    
    Returns:
        int: 0 if it never diverges, otherwise the iteration number where it diverged
    """
    z = 0 + 0j  # Start at zero
    for n in range(1, max_iter + 1):
        z = z * z + c
        if abs(z) > 2:  # (Divergence condition)
            return n
    return 0  # (if it never diverges)



