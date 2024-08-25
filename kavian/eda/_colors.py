import matplotlib.colors as mcolors

# https://claude.ai/chat/8446fe9a-55b1-4823-86e8-190769337e51

def color_thresholded_column(val, low_threshold, high_threshold):
    white_color = '#fff'  # White
    yellow_color = '#FBEC5D'  # Yellow
    red_color = '#e85440'  # Red

    if val < low_threshold:
        # White to Yellow range
        intensity = max(min(val / low_threshold, 1), 0)
        start_color = mcolors.to_rgb(white_color)
        end_color = mcolors.to_rgb(yellow_color)
    elif val < high_threshold:
        # Yellow to Red range
        intensity = max(min((val - low_threshold) / (high_threshold - low_threshold), 1), 0)
        start_color = mcolors.to_rgb(yellow_color)
        end_color = mcolors.to_rgb(red_color)
    else:
        # Full Red
        return f'color: {red_color}'

    # Interpolate between start and end colors
    rgb = tuple(start + intensity * (end - start) for start, end in zip(start_color, end_color))
    color = mcolors.to_hex(rgb)

    return f'color: {color}'


def color_adversarial_column(val, threshold):
    neg_color = '#e85440'  # Red
    pos_color = '#17aab5'  # Blue

    # Calculate the intensity based on the absolute value
    reg = 0.7
    intensity = min(abs(val) / reg, 1)

    rgb = mcolors.to_rgb(neg_color) if val < threshold else mcolors.to_rgb(pos_color)
    color = mcolors.to_hex([1 - intensity * (1 - r) for r in rgb])

    return f'color: {color}'



