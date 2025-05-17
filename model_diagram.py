import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Helper class for 3D Arrows (using FancyArrowPatch)
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None): # Compatible with Matplotlib 3.5+
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def draw_cuboid(ax, center_pos, size, facecolors='cyan', edgecolors='gray', alpha=0.7, label_text="", text_color='black', fontsize=7):
    cx, cy, cz = center_pos
    dx, dy, dz = size
    x, y, z = cx - dx/2, cy - dy/2, cz - dz/2
    verts = [
        (x, y, z), (x + dx, y, z), (x + dx, y + dy, z), (x, y + dy, z),
        (x, y, z + dz), (x + dx, y, z + dz), (x + dx, y + dy, z + dz), (x, y + dy, z + dz)
    ]
    faces = [
        [verts[0], verts[1], verts[2], verts[3]], [verts[4], verts[5], verts[6], verts[7]],
        [verts[0], verts[1], verts[5], verts[4]], [verts[2], verts[3], verts[7], verts[6]],
        [verts[1], verts[2], verts[6], verts[5]], [verts[0], verts[3], verts[7], verts[4]]
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=facecolors, linewidths=1, edgecolors=edgecolors, alpha=alpha))
    if label_text:
        ax.text(cx, cy, cz + dz/2 + 0.1*dz, label_text, ha='center', va='bottom', color=text_color, fontsize=fontsize, zorder=100,
                bbox=dict(boxstyle="round,pad=0.2", fc="ivory", alpha=0.6, ec='grey'))

def draw_arrow(ax, start_pos, end_pos, color='black', mutation_scale=15, arrowstyle='-|>', linestyle='solid'):
    arrow = Arrow3D([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]],
                    mutation_scale=mutation_scale, lw=1, arrowstyle=arrowstyle, color=color, linestyle=linestyle)
    ax.add_artist(arrow)

fig = plt.figure(figsize=(22, 15)) # Adjusted for potentially taller layout
ax = fig.add_subplot(111, projection='3d')

block_depth = 0.8
block_width = 2.0
block_height_std = 1.0

color_input = 'lightgreen'
color_sent_transformer = 'mediumseagreen'
color_fc = 'cornflowerblue'
color_relu = 'khaki'
color_reshape = 'lightgrey'
color_conv_transpose = 'salmon'
color_conv_block = 'steelblue'
color_tanh = 'mediumpurple'
color_codebook = 'lightcoral'
color_output_final = 'darkolivegreen'

x_curr = 0
y_center_main_path = 0 # Y-depth for the main horizontal path
z_level_main_path = block_depth / 2 # Z-level for the main horizontal path

all_plot_x_coords = []
all_plot_y_coords = [y_center_main_path]
all_plot_z_coords = [z_level_main_path]

# 1. Input Text
label_input = "lightgreen block 1"
pos_input = (x_curr, y_center_main_path, z_level_main_path)
draw_cuboid(ax, pos_input, (block_height_std*0.8, block_width*0.8, block_depth), color_input, label_text=label_input)
all_plot_x_coords.append(pos_input[0])
prev_block_output_connector = (pos_input[0] + (block_height_std*0.8)*0.5, pos_input[1], pos_input[2])
x_curr += block_height_std * 2.5

# 2. SentenceTransformer
label_st = "mediumseagreen block 1"
pos_st = (x_curr, y_center_main_path, z_level_main_path)
draw_cuboid(ax, pos_st, (block_height_std*1.5, block_width, block_depth), color_sent_transformer, label_text=label_st)
draw_arrow(ax, prev_block_output_connector, (pos_st[0] - (block_height_std*1.5)*0.5, pos_st[1], pos_st[2]))
all_plot_x_coords.append(pos_st[0])
prev_block_output_connector = (pos_st[0] + (block_height_std*1.5)*0.5, pos_st[1], pos_st[2])
x_curr += block_height_std * 3.0

# 3. FC Layers
# FC1
label_fc1 = "cornflowerblue block 1"
pos_fc1 = (x_curr, y_center_main_path, z_level_main_path)
draw_cuboid(ax, pos_fc1, (block_height_std, block_width, block_depth), color_fc, label_text=label_fc1)
draw_arrow(ax, prev_block_output_connector, (pos_fc1[0] - block_height_std*0.5, pos_fc1[1], pos_fc1[2]))
all_plot_x_coords.append(pos_fc1[0])
prev_block_output_connector_fc = (pos_fc1[0] + block_height_std*0.5, pos_fc1[1], pos_fc1[2])
x_curr_fc = x_curr + block_height_std * 2.0

# ReLU
label_relu1 = "khaki block 1"
pos_relu1 = (x_curr_fc, y_center_main_path, z_level_main_path)
draw_cuboid(ax, pos_relu1, (block_height_std*0.5, block_width*0.8, block_depth*0.8), color_relu, label_text=label_relu1)
draw_arrow(ax, prev_block_output_connector_fc, (pos_relu1[0] - (block_height_std*0.5)*0.5, pos_relu1[1], pos_relu1[2]))
all_plot_x_coords.append(pos_relu1[0])
prev_block_output_connector_fc = (pos_relu1[0] + (block_height_std*0.5)*0.5, pos_relu1[1], pos_relu1[2])
x_curr_fc += block_height_std * 1.75

# FC2
label_fc2 = "cornflowerblue block 2"
pos_fc2 = (x_curr_fc, y_center_main_path, z_level_main_path)
draw_cuboid(ax, pos_fc2, (block_height_std, block_width, block_depth), color_fc, label_text=label_fc2)
draw_arrow(ax, prev_block_output_connector_fc, (pos_fc2[0] - block_height_std*0.5, pos_fc2[1], pos_fc2[2]))
all_plot_x_coords.append(pos_fc2[0])
prev_block_output_connector = (pos_fc2[0] + block_height_std*0.5, pos_fc2[1], pos_fc2[2]) # Output of FC2
x_curr = x_curr_fc + block_height_std * 2.5

# 4. Reshape (conceptual)
label_reshape = "lightgrey block 1"
pos_reshape = (x_curr, y_center_main_path, z_level_main_path)
draw_cuboid(ax, pos_reshape, (block_height_std*0.7, block_width*0.7, block_depth*0.7), color_reshape, label_text=label_reshape, fontsize=6)
draw_arrow(ax, prev_block_output_connector, (pos_reshape[0] - (block_height_std*0.7)*0.5, pos_reshape[1], pos_reshape[2]))
all_plot_x_coords.append(pos_reshape[0])
# Connector for the first ladder rung, from Reshape output
current_ladder_input_connector = (pos_reshape[0] + (block_height_std*0.7)*0.5, pos_reshape[1], pos_reshape[2])

# Initialize counters for block colors
color_counters = {
    "salmon": 0,
    "steelblue": 0,
    "mediumpurple": 0,
    "lightcoral": 0,
}

# 5. Upsampling Blocks (Ladder Structure going UP in Z)
upblock_configs = [
    {"in_c": 512, "out_c_tconv": 512, "out_c_conv": 512, "id":0, "spatial_factor": "2H, 2W"},
    {"in_c": 512, "out_c_tconv": 256, "out_c_conv": 256, "id":1, "spatial_factor": "4H, 4W"},
    {"in_c": 256, "out_c_tconv": 128, "out_c_conv": 128, "id":2, "spatial_factor": "8H, 8W"},
    {"in_c": 128, "out_c_tconv": 64,  "out_c_conv": 64,  "id":3, "spatial_factor": "16H, 16W", "final_act": "Tanh"}
]

# Ladder positioning parameters
x_ladder_rung_start_x_base = x_curr + block_height_std * 0.7/2 + block_height_std * 2.5 # Base X for the TConv of the first rung (i=0)
overall_x_displacement_per_rung = block_height_std * 3.0  # INCREASED POSITIVE: Slants rungs to the right (positive X)
spacing_tconv_to_conv = block_height_std * 2.5           # X-spacing between TConv and Conv/Tanh within a rung
spacing_conv_to_codebook = block_height_std * 2.0        # NEW: X-spacing between Conv/Tanh and Codebook

y_ladder_depth = y_center_main_path # Ladder rungs at the same depth as main path

z_ladder_current_level = z_level_main_path # Start ladder at the same Z as Reshape output
vertical_spacing_ladder_z = block_depth + block_height_std * 1.5 # How much each rung goes up in Z

all_codebook_centers_for_output = []

# Store the output connector of the last block in the main path (Reshape)
# This will be the input for the very first arrow to the ladder.
initial_ladder_input_connector = current_ladder_input_connector

for i, conf in enumerate(upblock_configs):
    if i > 0: # For rungs after the first, move current Z level up
        z_ladder_current_level += vertical_spacing_ladder_z
    all_plot_z_coords.append(z_ladder_current_level)

    # Calculate X positions for the current rung's components
    current_rung_overall_x_offset = i * overall_x_displacement_per_rung
    
    actual_x_tconv_center = x_ladder_rung_start_x_base + current_rung_overall_x_offset
    actual_x_conv_center = actual_x_tconv_center + spacing_tconv_to_conv
    actual_x_codebook_center = actual_x_conv_center + spacing_conv_to_codebook # MODIFIED: Codebook X forward of Conv/Tanh

    all_plot_x_coords.append(actual_x_tconv_center)

    # ConvTranspose2d
    color_counters["salmon"] += 1
    label_tconv = f"salmon block {color_counters['salmon']}"
    pos_tconv = (actual_x_tconv_center, y_ladder_depth, z_ladder_current_level)
    draw_cuboid(ax, pos_tconv, (block_height_std, block_width, block_depth), color_conv_transpose, label_text=label_tconv)
    
    # Define the target connection point for TConv input (center of its left face for i=0, bottom-center for i>0)
    if i == 0:
        tconv_input_connection_point = (pos_tconv[0] - block_height_std*0.5, pos_tconv[1], pos_tconv[2]) # Center of left face
    else:
        tconv_input_connect_z = pos_tconv[2] - block_depth*0.5 # Z-coordinate of the bottom face
        # MODIFIED: Connect to the center of the bottom face
        tconv_input_connection_point = (pos_tconv[0], pos_tconv[1], tconv_input_connect_z) # Center of bottom face

    # Arrow logic for U-Net skip connection style
    if i == 0:
        # Direct arrow from Reshape output to the first TConv input (center of left face)
        draw_arrow(ax, initial_ladder_input_connector, tconv_input_connection_point)
    else:
        # current_ladder_input_connector is output of previous Conv/Tanh
        # MODIFIED: Single direct arrow to the bottom-center of the TConv input face
        draw_arrow(ax, current_ladder_input_connector, tconv_input_connection_point)
    
    # Conv Block / Tanh
    pos_conv = (actual_x_conv_center, y_ladder_depth, z_ladder_current_level)
    all_plot_x_coords.append(actual_x_conv_center)

    if "final_act" not in conf:
        color_counters["steelblue"] += 1
        label_conv = f"steelblue block {color_counters['steelblue']}"
        color_seq = color_conv_block
    else:
        color_counters["mediumpurple"] += 1
        label_conv = f"mediumpurple block {color_counters['mediumpurple']}"
        color_seq = color_tanh
    draw_cuboid(ax, pos_conv, (block_height_std, block_width, block_depth), color_seq, label_text=label_conv)
    
    # Arrow from TConv output (center of right face) to this Conv/Tanh input (center of left face)
    arrow_start_tconv_out = (pos_tconv[0] + block_height_std*0.5, pos_tconv[1], pos_tconv[2])
    arrow_end_conv_in = (pos_conv[0] - block_height_std*0.5, pos_conv[1], pos_conv[2])
    draw_arrow(ax, arrow_start_tconv_out, arrow_end_conv_in, arrowstyle='->') # Ensure single-sided
    
    # Codebook for this stage
    color_counters["lightcoral"] += 1
    cb_label = f"lightcoral block {color_counters['lightcoral']}"
    # Position Codebook: X forward of Conv/Tanh, Y aligned with ladder, Z is same as Conv/Tanh
    cb_pos = (actual_x_codebook_center, y_ladder_depth, z_ladder_current_level) # MODIFIED: Y aligned with ladder
    draw_cuboid(ax, cb_pos, (block_height_std*0.8, block_width*0.7, block_depth*0.8), color_codebook, label_text=cb_label)
    all_codebook_centers_for_output.append(cb_pos)
    all_plot_x_coords.append(cb_pos[0]) 
    
    # Arrow from Conv/Tanh block output (center of RIGHT/FORWARD X face) to Codebook input (center of LEFT/BACKWARD X face)
    arrow_start_cb = (pos_conv[0] + block_height_std*0.5, pos_conv[1], pos_conv[2]) # Center of right X face of Conv/Tanh
    arrow_end_cb = (cb_pos[0] - (block_height_std*0.8)*0.5, cb_pos[1], cb_pos[2])   # Center of left X face of Codebook
    draw_arrow(ax, arrow_start_cb, arrow_end_cb, color='dimgray', arrowstyle='->', linestyle='solid', mutation_scale=10)

    # Update input connector for the *next* TConv block (output of current Conv/Tanh block)
    current_ladder_input_connector = (pos_conv[0] + block_height_std*0.5, pos_conv[1], pos_conv[2])

# --- Final Output Indication ---
# Adjust output zone based on the new codebook Y positions if necessary
if all_codebook_centers_for_output:
    output_zone_y = all_codebook_centers_for_output[0][1] # Align Y with the first codebook's Y
    output_zone_z = all_codebook_centers_for_output[0][2] # Align Z with the first codebook's Z (should be same for all on this plane)
else: # Fallback if no codebooks somehow
    output_zone_y = y_center_main_path
    output_zone_z = z_level_main_path

output_zone_x = max(all_plot_x_coords) + block_height_std * 2.5 
all_plot_x_coords.append(output_zone_x)

label_output_text = "4 Outputs"
pos_output_text_display = (output_zone_x, output_zone_y, output_zone_z + block_width*0.3) 
ax.text(pos_output_text_display[0], pos_output_text_display[1], pos_output_text_display[2], label_output_text,
        ha='center', va='center', color=color_output_final, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.8))

for cb_center_pos in all_codebook_centers_for_output:
    start_arrow_output = (cb_center_pos[0] + (block_height_std*0.8)*0.5, cb_center_pos[1], cb_center_pos[2])
    end_arrow_output = (output_zone_x - block_height_std*0.5, output_zone_y, output_zone_z) 
    draw_arrow(ax, start_arrow_output, end_arrow_output, color=color_codebook, arrowstyle='-|>')

# --- Plot Settings ---
unique_x = sorted(list(set(all_plot_x_coords)))
unique_y = sorted(list(set(all_plot_y_coords)))
unique_z = sorted(list(set(all_plot_z_coords)))

min_x, max_x = min(unique_x) - block_height_std * 1.0, max(unique_x) + block_height_std * 1.0
min_y, max_y = min(unique_y) - block_width * 1.0, max(unique_y) + block_width * 1.0
min_z, max_z = min(unique_z) - block_depth * 1.0, max(unique_z) + block_depth * 1.5 # Extra space for Z

ax.set_xlim([min_x, max_x])
ax.set_ylim([min_y, max_y])
ax.set_zlim([min_z, max_z])

ax.set_title("PyTorch Model Visualization (3D Style)", fontsize=16)
ax.grid(False)
ax.set_axis_off()
ax.view_init(elev=0, azim=-90) 
plt.tight_layout()
plt.savefig("model_visualization.png", dpi=300)
print("Model visualization saved to model_visualization.png")

