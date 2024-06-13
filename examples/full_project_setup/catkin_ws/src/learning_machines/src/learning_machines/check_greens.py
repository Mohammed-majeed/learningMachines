def count_green_pixels_and_percentage(image):
    "Takes in an image, returns the number of green pixels, and the percentage of green pixels in that image"
    # Define the range for green in BGR 
    lower_bound = np.array([0, 125, 0])
    upper_bound = np.array([50, 255, 50])

    # Get the dimensions of the image
    height, width, _ = image.shape

    #Total number of pixels in image
    number_of_pixels = ((height*width) // 3)
    
    # Create a mask for green pixels
    green_mask = (lower_bound <= image) & (image <= upper_bound)
    
    # Ensure the mask is applied correctly across the color channels
    green_mask = np.all(green_mask, axis=2)

    green_count = np.sum(green_mask)

    #Calculate percentages
    pct_green  = (green_count / number_of_pixels) *100

    return green_count, pct_green



def get_green_percentages(image):
    "Takes in an image, divides it into three blocks, left right and middle, and returns the percentage of pixels that are green in each block as a dict"
    # Get the dimensions of the image
    height, width, _ = image.shape
    
    # Calculate the width of each block
    block_width = width // 3

    #Number of pixels in a block
    number_of_pixels = ((height*width) // 3)
    
    blocks = {
    
    "left_block" : image[:, :block_width],
    "middle_block" : image[:, block_width:2*block_width],
    "right_block" : image[:, 2*block_width:],
    }

    percentages ={}
    #Print out values, save images and store percentages in dict
    for name, img in blocks.items():
        green_count, green_percentage = count_green_pixels_and_percentage(img)
        print(f"Number of green pixels in {name}", green_count)
        print(f"Percent of green pixels in {name}", green_percentage)
        cv2.imwrite(str(FIGRURES_DIR / f"photo_{name}.png"), img)
        percentages[name] = green_percentage

    return percentages
    #Check number of green pixels