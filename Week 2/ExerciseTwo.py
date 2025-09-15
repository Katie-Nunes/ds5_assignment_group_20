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


def map_pixel_to_complex(i: int,j: int, width: int, x_range: tuple(float,float) = (-1.5,0.5),y_range: tuple(float,float) = (-1,1)) -> complex:
    ## Maps a pixel coordinate (i,j) to a point in the complex plane
    
	x_min,x_max = x_range
	y_min,y_max = y_range
	x = x_min + (j / width) * (x_max - x_min)
	y = y_min + (i / width) * (y_max - y_min)
	return complex(x,y)