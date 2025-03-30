import cv2
import numpy as np

def draw_variable_bleed_lines(overlay, width, height, rolls, border_height, film_height, roll_spacing, 
                             min_bleed_mm=1.0, max_bleed_mm=1.75, mm_to_px=1.0,
                             base_period=3, period_variation=0.5, amplitude_variation=0.2):
    """
    Draw white bleed lines with variable height at the top of the top roll and bottom of the bottom roll.
    
    Parameters:
        overlay: The image array to draw on
        width, height: Dimensions of the image
        rolls: Number of film rolls
        border_height: Height of the border in pixels
        film_height: Height of each film roll in pixels
        roll_spacing: Spacing between rolls in pixels
        min_bleed_mm, max_bleed_mm: Range of bleed line heights in mm
        mm_to_px: Conversion factor from mm to pixels
        base_period: Base period of sine wave as fraction of image width
        period_variation: Variation in period (0-1)
        amplitude_variation: Variation in amplitude (0-1)
    """
    # Convert bleed line heights from mm to pixels
    min_bleed_px = int(min_bleed_mm * mm_to_px)
    max_bleed_px = int(max_bleed_mm * mm_to_px)
    base_bleed = (min_bleed_px + max_bleed_px) // 2
    max_amplitude = (max_bleed_px - min_bleed_px) // 2
    
    # Top roll position
    top_roll_top = border_height
    
    # Bottom roll position
    bottom_roll_bottom = height - border_height
    
    # Draw variable height bleed line at top of top roll
    for x in range(width):
        # Create variable period and amplitude
        x_normalized = x / width
        period = base_period * (1 + period_variation * np.sin(2 * np.pi * x_normalized * 1.7))
        amplitude = max_amplitude * (1 + amplitude_variation * np.sin(2 * np.pi * x_normalized * 2.3))
        
        # Calculate height using sine wave with variable period and amplitude
        top_height = int(base_bleed + amplitude * np.sin(2 * np.pi * x / (width / period)))
        
        # Ensure height is within bounds
        top_height = max(min_bleed_px, min(max_bleed_px, top_height))
        
        # Draw white line at top of top roll
        overlay[top_roll_top:top_roll_top + top_height, x, 0:3] = 255  # Set RGB channels to white
    
    # Draw variable height bleed line at bottom of bottom roll
    for x in range(width):
        # Create different variable period and amplitude for bottom
        x_normalized = x / width
        period = base_period * (1 + period_variation * np.sin(2 * np.pi * x_normalized * 2.1))
        amplitude = max_amplitude * (1 + amplitude_variation * np.sin(2 * np.pi * x_normalized * 1.9))
        
        # Calculate height using sine wave with variable period and amplitude
        bottom_height = int(base_bleed + amplitude * np.sin(2 * np.pi * x / (width / period) + 0.7))
        
        # Ensure height is within bounds
        bottom_height = max(min_bleed_px, min(max_bleed_px, bottom_height))
        
        # Draw white line at bottom of bottom roll
        overlay[bottom_roll_bottom - bottom_height:bottom_roll_bottom, x, 0:3] = 255  # Set RGB channels to white


def create_sprocket_overlay(width, height, rolls=4, corner_radius_mm=0.25, 
                           min_bleed_mm=0.75, max_bleed_mm=1.5):
    # Constants in pixels (converted from mm assuming 35mm film height)
    mm_to_px = height / (35 * rolls + 29)  # 35mm per roll + 10mm top/bottom border
    film_height = int(35 * mm_to_px)
    border_height = int(10 * mm_to_px)
    sprocket_width = int(1.9 * mm_to_px)
    sprocket_height = int(2.794 * mm_to_px)
    sprocket_spacing = int(4.7498 * mm_to_px)
    sprocket_offset = int(1.9 * mm_to_px)
    roll_spacing = int(3 * mm_to_px)  # Spacing between film rolls
    # Convert corner radius from mm to pixels
    corner_radius = int(corner_radius_mm * mm_to_px)
    total_film_and_borders = int((film_height * 4) + (border_height * 2) + (roll_spacing * 3))  # Height of the picture area
    
    # Create a blank image for the overlay with an alpha channel
    overlay = np.zeros((height, width, 4), dtype=np.uint8)  # RGBA: Black background with alpha
    
    # Create a separate layer for the bleed lines
    bleed_lines = np.zeros((height, width), dtype=np.uint8)
    
    # Extract the alpha channel for drawing and ensure it is contiguous
    alpha_channel = np.ascontiguousarray(overlay[:, :, 3])
    
    # Draw black borders
    alpha_channel[:border_height, :] = 255  # Top border (alpha channel)
    alpha_channel[-border_height:, :] = 255  # Bottom border (alpha channel)
    
    # Draw variable white bleed lines on the separate layer
    # Top roll position
    top_roll_top = border_height
    # Bottom roll position
    bottom_roll_bottom = height - border_height
    
    # Draw variable height bleed line at top of top roll
    for x in range(width):
        # Create variable period and amplitude
        x_normalized = x / width
        period = 3 * (1 + 0.5 * np.sin(2 * np.pi * x_normalized * 1.7))
        amplitude = (max_bleed_mm - min_bleed_mm) * mm_to_px / 2 * (1 + 0.2 * np.sin(2 * np.pi * x_normalized * 2.3))
        
        # Calculate height using sine wave with variable period and amplitude
        base_bleed = (min_bleed_mm + max_bleed_mm) / 2 * mm_to_px
        top_height = int(base_bleed + amplitude * np.sin(2 * np.pi * x / (width / period)))
        
        # Ensure height is within bounds
        top_height = max(int(min_bleed_mm * mm_to_px), min(int(max_bleed_mm * mm_to_px), top_height))
        
        # Draw white line at top of top roll
        bleed_lines[top_roll_top:top_roll_top + top_height, x] = 255
    
    # Draw variable height bleed line at bottom of bottom roll
    for x in range(width):
        # Create different variable period and amplitude for bottom
        x_normalized = x / width
        period = 3 * (1 + 0.5 * np.sin(2 * np.pi * x_normalized * 2.1))
        amplitude = (max_bleed_mm - min_bleed_mm) * mm_to_px / 2 * (1 + 0.2 * np.sin(2 * np.pi * x_normalized * 1.9))
        
        # Calculate height using sine wave with variable period and amplitude
        base_bleed = (min_bleed_mm + max_bleed_mm) / 2 * mm_to_px
        bottom_height = int(base_bleed + amplitude * np.sin(2 * np.pi * x / (width / period) + 0.7))
        
        # Ensure height is within bounds
        bottom_height = max(int(min_bleed_mm * mm_to_px), min(int(max_bleed_mm * mm_to_px), bottom_height))
        
        # Draw white line at bottom of bottom roll
        bleed_lines[bottom_roll_bottom - bottom_height:bottom_roll_bottom, x] = 255
    
    # Apply Gaussian blur to the bleed lines for smoother transitions
    bleed_lines = cv2.GaussianBlur(bleed_lines, (9, 9), 0)
    
    # Copy blurred bleed lines to RGB channels of overlay
    for c in range(3):
        overlay[:, :, c] = bleed_lines

    # Draw sprocket holes for each roll
    for roll in range(rolls):
        roll_top = border_height + roll * (film_height + roll_spacing)
        roll_bottom = roll_top + film_height

        # Draw black bar for roll spacing (except after the last roll)
        if roll < rolls - 1:
            spacing_top = roll_bottom
            spacing_bottom = spacing_top + roll_spacing
            alpha_channel[spacing_top:spacing_bottom, :] = 255  # Black bar for spacing (alpha channel)

        # Draw white bleed lines
    #    if roll == 0:
    #        alpha_channel[roll_top - 2:roll_top, :] = 255  # Top white bleed (alpha channel)
    #    if roll == rolls - 1:
    #        alpha_channel[roll_bottom - 2:roll_bottom, :] = 255  # Bottom white bleed (alpha channel)

        # Draw sprocket holes horizontally along the width
        for x in range(0, width, sprocket_spacing):
            # Define sprocket hole coordinates
            top_left = (x, roll_top + sprocket_offset)
            bottom_right = (x + sprocket_width, roll_top + sprocket_offset + sprocket_height)

            # Draw rounded corners first on the alpha channel
            # Top-left corner
            cv2.circle(alpha_channel, (top_left[0] + corner_radius, top_left[1] + corner_radius), corner_radius, 255, -1)
            # Top-right corner
            cv2.circle(alpha_channel, (bottom_right[0] - corner_radius, top_left[1] + corner_radius), corner_radius, 255, -1)
            # Bottom-left corner
            cv2.circle(alpha_channel, (top_left[0] + corner_radius, bottom_right[1] - corner_radius), corner_radius, 255, -1)
            # Bottom-right corner
            cv2.circle(alpha_channel, (bottom_right[0] - corner_radius, bottom_right[1] - corner_radius), corner_radius, 255, -1)

            # Draw the rectangle excluding the corners on the alpha channel
            cv2.rectangle(alpha_channel, (top_left[0] + corner_radius, top_left[1]),
                          (bottom_right[0] - corner_radius, bottom_right[1]), 255, -1)
            cv2.rectangle(alpha_channel, (top_left[0], top_left[1] + corner_radius),
                          (bottom_right[0], bottom_right[1] - corner_radius), 255, -1)

            # Repeat for bottom sprocket holes
            top_left = (x, roll_bottom - sprocket_offset - sprocket_height)
            bottom_right = (x + sprocket_width, roll_bottom - sprocket_offset)

            # Draw rounded corners first on the alpha channel
            # Top-left corner
            cv2.circle(alpha_channel, (top_left[0] + corner_radius, top_left[1] + corner_radius), corner_radius, 255, -1)
            # Top-right corner
            cv2.circle(alpha_channel, (bottom_right[0] - corner_radius, top_left[1] + corner_radius), corner_radius, 255, -1)
            # Bottom-left corner
            cv2.circle(alpha_channel, (top_left[0] + corner_radius, bottom_right[1] - corner_radius), corner_radius, 255, -1)
            # Bottom-right corner
            cv2.circle(alpha_channel, (bottom_right[0] - corner_radius, bottom_right[1] - corner_radius), corner_radius, 255, -1)

            # Draw the rectangle excluding the corners on the alpha channel
            cv2.rectangle(alpha_channel, (top_left[0] + corner_radius, top_left[1]),
                          (bottom_right[0] - corner_radius, bottom_right[1]), 255, -1)
            cv2.rectangle(alpha_channel, (top_left[0], top_left[1] + corner_radius),
                          (bottom_right[0], bottom_right[1] - corner_radius), 255, -1)

    # Apply Gaussian blur to the alpha channel for anti-aliasing
    overlay[:, :, 3] = cv2.GaussianBlur(alpha_channel, (5, 5), 0)  # Kernel size (5, 5) for smoothing

    return overlay

def apply_sprocket_overlay(source_image_path, output_image_path, width, height):
    # Load the source image
    source_image = cv2.imread(source_image_path)
    if source_image is None:
        raise ValueError("Image not found or invalid image path.")

    # Constants in pixels (recalculate to match create_sprocket_overlay)
    mm_to_px = height / (35 * 4 + 29)  # 35mm per roll + 10mm top/bottom border
    film_height = int(35 * mm_to_px)
    border_height = int(10 * mm_to_px)
    roll_spacing = int(3 * mm_to_px)

    # Resize the source image to fit within the film rolls
    source_height, source_width = source_image.shape[:2]
    aspect_ratio = width / height
    source_aspect_ratio = source_width / source_height

    if source_aspect_ratio > aspect_ratio:
        # Crop width to match the target aspect ratio
        new_width = int(source_height * aspect_ratio)
        crop_x = (source_width - new_width) // 2
        source_image = source_image[:, crop_x:crop_x + new_width]
    else:
        # Crop height to match the target aspect ratio
        new_height = int(source_width / aspect_ratio)
        crop_y = (source_height - new_height) // 2
        source_image = source_image[crop_y:crop_y + new_height, :]

    # Resize the cropped image to match the film area
    film_area_height = height - (2 * border_height) - (3 * roll_spacing)  # Exclude borders and spacing
    film_area_width = width
    resized_image = cv2.resize(source_image, (film_area_width, int(film_area_height)))

    # Convert the resized image to grayscale
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Create the sprocket overlay
    overlay = create_sprocket_overlay(width, height)

    # Create a result array with the same dimensions as the overlay
    result = np.ones((height, width), dtype=np.uint8) * 255  # White background

    # Place the resized image in the center of the overlay
    # Calculate roll heights based on the actual film height
    mm_to_px = height / (35 * 4 + 29)  # 35mm per roll + 10mm top/bottom border
    film_height = int(35 * mm_to_px)
    border_height = int(10 * mm_to_px)
    roll_spacing = int(3 * mm_to_px)
    
    # Calculate the vertical positions for each roll
    roll_positions = []
    for roll in range(4):
        roll_top = border_height + roll * (film_height + roll_spacing)
        roll_bottom = roll_top + film_height
        roll_positions.append((roll_top, roll_bottom))
    
    # Divide the resized image into 4 equal parts
    resized_height = resized_image.shape[0]
    section_height = resized_height // 4
    
    # Place each section onto the corresponding film roll
    for roll, (roll_top, roll_bottom) in enumerate(roll_positions):
        start_idx = roll * section_height
        end_idx = start_idx + (section_height - roll_spacing) if roll < 3 else resized_height
        
        section = resized_image[start_idx:end_idx]
        if roll_bottom - roll_top != section.shape[0]:
            # Resize the section to fit the roll height
            section = cv2.resize(section, (width, roll_bottom - roll_top))
        
        result[roll_top:roll_bottom, :] = section

    # Apply the alpha channel for blending
    alpha = overlay[:, :, 3] / 255.0  # Normalize alpha to range [0, 1]
    
    # Create a new result array with 3 channels
    final_result = np.zeros((height, width, 3), dtype=np.uint8)
    
    # First blend the grayscale image with black for sprocket holes
    for c in range(3):
        final_result[:, :, c] = (1 - alpha) * result + alpha * 0  # Blend with black
    
    # Then, overlay the white bleed lines on top with smooth blending
    # Create a normalized weight matrix from the RGB channels of the overlay
    white_weight = overlay[:, :, 0] / 255.0  # Use one channel (all are the same)
    
    # Apply weighted blend of white pixels from overlay
    for c in range(3):
        # Blend based on weight: result = (1-weight)*current + weight*white
        final_result[:, :, c] = (1 - white_weight) * final_result[:, :, c] + white_weight * 255
    
    # Save the result
    cv2.imwrite(output_image_path, final_result)

if __name__ == "__main__":
    # Example usage
    output_width = 6000  # Configurable width
    output_height = int(output_width / (400 / 115))  # Maintain aspect ratio
    source_image_path = "input.jpg"  # Replace with your input image path
    output_image_path = "output.jpg"  # Replace with your desired output path

    apply_sprocket_overlay(source_image_path, output_image_path, output_width, output_height)
