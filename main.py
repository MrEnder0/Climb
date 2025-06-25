import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.graph import route_through_array
from scipy.ndimage import gaussian_filter
import os

# --- Step 0: Initial Setup ---
def load_or_create_image():
    """Load image or create a demo mountain scene"""
    img_path = "C:/Users/CodeN/OneDrive/Desktop/peak4.png"
    
    try:
        if os.path.exists(img_path):
            img_color = cv2.imread(img_path)
            if img_color is not None:
                # Convert to RGB for processing and display
                img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
                print("Image loaded successfully!")
                print(f"Image shape: {img_color_rgb.shape}")
                return img_color_rgb
    except Exception as e:
        print(f"Error loading image: {e}")
    
    # Create a more realistic mountain scene for demonstration
    print("Creating a realistic demo mountain scene...")
    img_color_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Create terrain layers with noise for realism
    x = np.linspace(0, 1, 512)
    y = np.linspace(0, 1, 512)
    xv, yv = np.meshgrid(x, y)
    
    # Add some random noise for texture
    np.random.seed(42)  # For reproducible results
    noise = np.random.normal(0, 0.1, (512, 512))
    
    # Mountain elevation with multiple peaks
    mountain1 = np.exp(-((xv - 0.3)**2 + (yv - 0.7)**2) / 0.08) * 0.8
    mountain2 = np.exp(-((xv - 0.7)**2 + (yv - 0.6)**2) / 0.12) * 0.9
    mountain3 = np.exp(-((xv - 0.5)**2 + (yv - 0.4)**2) / 0.15) * 0.6
    
    # Combine mountains and add noise
    elevation_base = mountain1 + mountain2 + mountain3 + noise * 0.1
    elevation_base = np.clip(elevation_base, 0, 1)
    
    # Create realistic mountain colors
    # Sky gradient (blue at top)
    sky_gradient = np.linspace(1, 0, 512)[:, None]
    sky_mask = (elevation_base < 0.2) & (yv < 0.6)
    
    # Rock colors (gray/brown)
    rock_intensity = elevation_base * 0.7 + 0.3
    
    # Snow on high peaks
    snow_mask = elevation_base > 0.7
    
    # Apply colors
    img_color_rgb[:,:,0] = np.where(sky_mask, 135 + sky_gradient.squeeze() * 50, 
                                   np.where(snow_mask, 240, rock_intensity * 120 + 60)).astype(np.uint8)
    img_color_rgb[:,:,1] = np.where(sky_mask, 206 + sky_gradient.squeeze() * 30,
                                   np.where(snow_mask, 248, rock_intensity * 100 + 40)).astype(np.uint8)
    img_color_rgb[:,:,2] = np.where(sky_mask, 235 + sky_gradient.squeeze() * 20,
                                   np.where(snow_mask, 255, rock_intensity * 80 + 30)).astype(np.uint8)
    
    return img_color_rgb

# Load or create image
img_color_rgb = load_or_create_image()

# --- Step 1: Improved Sky Detection ---
def detect_sky_robust(img_rgb):
    """Robust sky detection using multiple methods"""
    h, w = img_rgb.shape[:2]
    
    # Convert to different color spaces
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Method 1: Blue sky detection
    blue_mask = (img_rgb[:,:,2] > img_rgb[:,:,0]) & (img_rgb[:,:,2] > img_rgb[:,:,1])
    blue_dominant = img_rgb[:,:,2] > np.mean(img_rgb[:,:,2]) + np.std(img_rgb[:,:,2]) * 0.5
    
    # Method 2: Brightness-based (sky is usually bright)
    bright_mask = gray > np.percentile(gray, 85)
    
    # Method 3: Position-based (sky is in upper regions)
    position_mask = np.zeros((h, w), dtype=bool)
    position_mask[:h//3, :] = True  # Top third more likely to be sky
    
    # Method 4: Color uniformity (sky has less texture)
    kernel = np.ones((7,7), np.float32) / 49
    smooth_gray = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    texture_var = np.abs(gray.astype(np.float32) - smooth_gray)
    low_texture = texture_var < np.percentile(texture_var, 30)
    
    # Method 5: HSV-based sky detection
    hue = img_hsv[:,:,0]
    sat = img_hsv[:,:,1]
    val = img_hsv[:,:,2]
    
    # Sky characteristics in HSV
    sky_hue = ((hue >= 90) & (hue <= 130)) | (hue <= 10) | (hue >= 170)  # Blue or white
    sky_sat = sat < 80  # Low saturation
    sky_val = val > 150  # High value
    
    # Combine all methods with weights
    sky_score = (
        blue_mask.astype(float) * 0.15 +
        blue_dominant.astype(float) * 0.15 +
        bright_mask.astype(float) * 0.2 +
        position_mask.astype(float) * 0.2 +
        low_texture.astype(float) * 0.1 +
        (sky_hue & sky_sat & sky_val).astype(float) * 0.2
    )
    
    # Apply threshold - make it less aggressive
    sky_mask = sky_score > 0.5  # Increased threshold to be more conservative
    
    # Clean up using morphological operations
    kernel = np.ones((5,5), np.uint8)
    sky_mask = cv2.morphologyEx(sky_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
    
    # Remove small sky regions (noise)
    contours, _ = cv2.findContours(sky_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 1000:  # Remove small areas
            cv2.fillPoly(sky_mask, [contour], 0)
    
    return sky_mask > 0

def analyze_terrain_advanced(img_rgb):
    """Advanced terrain analysis with better sky detection"""
    
    # Convert to different color spaces
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    
    # Detect sky areas
    sky_mask = detect_sky_robust(img_rgb)
    landscape_mask = ~sky_mask
    
    # Create elevation map only from landscape areas
    luminance = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # Use color saturation and hue to enhance depth perception
    saturation = img_hsv[:,:,1].astype(np.float32) / 255.0
    hue = img_hsv[:,:,0].astype(np.float32) / 179.0
    
    # Enhanced elevation map
    color_depth = luminance * 0.7 + saturation * 0.2 + (1 - hue) * 0.1
    
    # Set sky areas to very low elevation
    color_depth[sky_mask] = 0.0
    
    # Smooth the elevation map
    elevation = gaussian_filter(color_depth, sigma=2.0)
    
    return elevation, landscape_mask, img_hsv, img_lab

def detect_rock_areas(img_rgb, img_hsv, landscape_mask):
    """Detect rocky areas within landscape only"""
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Only analyze landscape areas
    gray_landscape = gray.copy()
    gray_landscape[~landscape_mask] = 0
    
    # Texture analysis
    kernel = np.ones((5,5), np.float32) / 25
    texture_smooth = cv2.filter2D(gray_landscape.astype(np.float32), -1, kernel)
    texture_measure = np.abs(gray_landscape.astype(np.float32) - texture_smooth)
    
    # Color-based rock detection
    saturation = img_hsv[:,:,1]
    value = img_hsv[:,:,2]
    
    # Rock characteristics: lower saturation, medium value, within landscape
    rock_mask = (saturation < 60) & (value > 80) & (value < 180) & landscape_mask
    
    # Edge detection for rocky areas
    edges = cv2.Canny(gray, 30, 100)
    rock_edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    rock_edges = rock_edges & landscape_mask  # Only within landscape
    
    # Combine detections
    texture_threshold = np.percentile(texture_measure[landscape_mask], 70) if np.any(landscape_mask) else 0
    rock_areas = ((texture_measure > texture_threshold) | rock_mask | (rock_edges > 0)) & landscape_mask
    
    # Clean up the mask
    kernel = np.ones((3,3), np.uint8)
    rock_areas = cv2.morphologyEx(rock_areas.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    return rock_areas > 0

# Analyze the terrain with improved detection
elevation, landscape_mask, img_hsv, img_lab = analyze_terrain_advanced(img_color_rgb)
rock_areas = detect_rock_areas(img_color_rgb, img_hsv, landscape_mask)

print(f"Detected {np.sum(rock_areas)} rocky pixels ({np.sum(rock_areas)/rock_areas.size*100:.1f}% of image)")
print(f"Detected {np.sum(~landscape_mask)} sky pixels ({np.sum(~landscape_mask)/landscape_mask.size*100:.1f}% of image)")
print(f"Landscape coverage: {np.sum(landscape_mask)/landscape_mask.size*100:.1f}% of image")

# --- Step 2: Calculate Gradients and Steepness ---
grad_x = cv2.Sobel(elevation, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(elevation, cv2.CV_64F, 0, 1, ksize=3)
steepness = np.sqrt(grad_x**2 + grad_y**2)

# --- Step 3: Find Horizon Target (Center of Horizon) ---
def find_horizon_target(landscape_mask, elevation):
    """Find a target point at the center of the horizon line"""
    
    h, w = landscape_mask.shape
    center_col = w // 2
    
    # Find the horizon line at the center column
    # Look from top to bottom for the first landscape pixel
    horizon_row = None
    for row in range(h):
        if landscape_mask[row, center_col]:
            horizon_row = row
            break
    
    if horizon_row is None:
        # Fallback: use middle of image
        horizon_row = h // 3
        print(f"Warning: No horizon found at center, using fallback position")
    
    # Try to find a good target point near the horizon
    # Look for an area with reasonable elevation
    search_radius = 30
    best_target = (horizon_row, center_col)
    best_elevation = 0
    
    for dr in range(-search_radius, search_radius + 1):
        for dc in range(-search_radius, search_radius + 1):
            test_row = max(0, min(h-1, horizon_row + dr))
            test_col = max(0, min(w-1, center_col + dc))
            
            if landscape_mask[test_row, test_col]:
                test_elevation = elevation[test_row, test_col]
                if test_elevation > best_elevation:
                    best_elevation = test_elevation
                    best_target = (test_row, test_col)
    
    print(f"Horizon target found at {best_target} with elevation {elevation[best_target]:.3f}")
    
    return best_target

# Find the horizon target
horizon_target = find_horizon_target(landscape_mask, elevation)

# --- Step 4: Find 3 Different Start Points ---
def find_three_start_points(landscape_mask, elevation):
    """Find 3 different starting points at the bottom"""
    
    h, w = landscape_mask.shape
    start_points = []
    
    # Define 3 search columns: left, center, right
    search_cols = [w//4, w//2, 3*w//4]
    
    for target_col in search_cols:
        # Search from bottom up to find landscape
        for row in range(h - 1, max(h - 50, 0), -1):
            candidates = []
            search_width = 30
            
            for offset in range(-search_width, search_width + 1):
                col = target_col + offset
                if 0 <= col < w and landscape_mask[row, col]:
                    candidates.append((row, col, elevation[row, col]))
            
            if candidates:
                # Choose the candidate with lowest elevation (valley/base)
                best_candidate = min(candidates, key=lambda x: x[2])
                start_points.append((best_candidate[0], best_candidate[1]))
                break
    
    # Remove any that are too close together
    filtered_starts = []
    min_distance = 80
    
    for start in start_points:
        too_close = False
        for existing in filtered_starts:
            distance = np.sqrt((start[0] - existing[0])**2 + (start[1] - existing[1])**2)
            if distance < min_distance:
                too_close = True
                break
        
        if not too_close:
            filtered_starts.append(start)
    
    # Ensure we have exactly 3 points
    while len(filtered_starts) < 3:
        # Add more points if needed
        for col in range(0, w, w//10):
            for row in range(h - 30, h):
                if landscape_mask[row, col]:
                    new_point = (row, col)
                    too_close = False
                    for existing in filtered_starts:
                        distance = np.sqrt((new_point[0] - existing[0])**2 + (new_point[1] - existing[1])**2)
                        if distance < min_distance:
                            too_close = True
                            break
                    
                    if not too_close:
                        filtered_starts.append(new_point)
                        break
                
                if len(filtered_starts) >= 3:
                    break
            if len(filtered_starts) >= 3:
                break
    
    # Take only the first 3
    filtered_starts = filtered_starts[:3]
    
    print(f"Found 3 start points:")
    for i, (row, col) in enumerate(filtered_starts):
        print(f"  Start {i+1}: ({row}, {col}) elevation: {elevation[row, col]:.3f}")
    
    return filtered_starts

start_points = find_three_start_points(landscape_mask, elevation)

# --- Step 5: Improved Cost Function ---
def create_climbing_cost_grid(elevation, steepness, rock_areas, landscape_mask):
    """Create a cost grid that prevents sky traversal but allows reaching horizon"""
    
    # Normalize inputs
    steepness_norm = steepness / (np.max(steepness) + 1e-8)
    elevation_norm = (elevation - np.min(elevation)) / (np.max(elevation) - np.min(elevation) + 1e-8)
    
    # Base cost components
    base_cost = 1.0
    steepness_cost = steepness_norm * 4.0  # Moderate penalty for steep areas
    rock_penalty = rock_areas.astype(float) * 6.0  # Rock penalty
    
    # Small preference for gaining elevation (encourages going up)
    elevation_cost = (1.0 - elevation_norm) * 0.3
    
    # Combine all costs for landscape areas
    total_cost = base_cost + steepness_cost + rock_penalty + elevation_cost
    
    # Make sky areas very expensive but not impossible (for edge cases)
    total_cost[~landscape_mask] = 10000.0  # Very high but finite cost
    
    # Ensure no NaN values
    total_cost = np.nan_to_num(total_cost, nan=10000.0, posinf=10000.0, neginf=1.0)
    
    return total_cost

cost_grid = create_climbing_cost_grid(elevation, steepness, rock_areas, landscape_mask)

# --- Step 6: Find 3 Paths to the Horizon Target ---
def find_three_paths_to_horizon(cost_grid, start_points, horizon_target, landscape_mask):
    """Find 3 different paths to the horizon target"""
    
    paths = []
    path_info = []
    
    colors = ['lime', 'red', 'cyan']
    path_names = ['Left Route', 'Center Route', 'Right Route']
    
    for i, start in enumerate(start_points):
        print(f"\nFinding path {i+1} from {start} to {horizon_target}")
        
        # Verify start point is in landscape
        if not landscape_mask[start[0], start[1]]:
            print(f"Start point {i+1} is not in landscape, adjusting...")
            # Find nearest landscape pixel
            h, w = landscape_mask.shape
            for radius in range(1, 20):
                found = False
                for dr in range(-radius, radius+1):
                    for dc in range(-radius, radius+1):
                        new_row = max(0, min(h-1, start[0] + dr))
                        new_col = max(0, min(w-1, start[1] + dc))
                        if landscape_mask[new_row, new_col]:
                            start = (new_row, new_col)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        # Verify target is in landscape
        if not landscape_mask[horizon_target[0], horizon_target[1]]:
            print(f"Horizon target is not in landscape! This shouldn't happen.")
            continue
        
        try:
            path, cost = route_through_array(
                cost_grid,
                start=start,
                end=horizon_target,
                fully_connected=True,
                geometric=True
            )
            
            if path is not None and len(path) > 1:
                paths.append(path)
                
                # Calculate path metrics
                path_array = np.array(path)
                path_elevations = elevation[path_array[:, 0], path_array[:, 1]]
                path_steepness = steepness[path_array[:, 0], path_array[:, 1]]
                path_rocks = rock_areas[path_array[:, 0], path_array[:, 1]]
                
                total_elevation_gain = np.sum(np.maximum(0, np.diff(path_elevations)))
                avg_steepness = np.mean(path_steepness)
                rock_encounters = np.sum(path_rocks)
                path_length = len(path)
                
                # Check if path actually reaches the target
                final_point = path[-1]
                distance_to_target = np.sqrt((final_point[0] - horizon_target[0])**2 + (final_point[1] - horizon_target[1])**2)
                reached_target = distance_to_target < 5  # Within 5 pixels
                
                info = {
                    'name': path_names[i],
                    'cost': cost,
                    'elevation_gain': total_elevation_gain,
                    'avg_steepness': avg_steepness,
                    'rock_encounters': rock_encounters,
                    'path_length': path_length,
                    'color': colors[i],
                    'start_point': start,
                    'reached_target': reached_target,
                    'final_elevation': path_elevations[-1],
                    'target_elevation': elevation[horizon_target]
                }
                path_info.append(info)
                
                print(f"✓ {path_names[i]}: Cost {cost:.1f}, Length {path_length}, Elevation gain {total_elevation_gain:.3f}")
                print(f"  Reached target: {'YES' if reached_target else 'NO'} (distance: {distance_to_target:.1f})")
                        
        except Exception as e:
            print(f"✗ Error finding path {i+1}: {e}")
    
    return paths, path_info

# Find the 3 paths
all_paths, path_info = find_three_paths_to_horizon(cost_grid, start_points, horizon_target, landscape_mask)

print(f"\nSuccessfully found {len(all_paths)} climbing routes to the horizon target")

# --- Step 7: Enhanced Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Main path visualization
ax1 = axes[0, 0]
ax1.imshow(img_color_rgb)

# Plot sky overlay (light blue tint)
sky_overlay = np.zeros_like(img_color_rgb)
sky_overlay[~landscape_mask] = [100, 150, 255]  # Light blue for sky
ax1.imshow(sky_overlay, alpha=0.15)

# Plot rock areas with transparency
if np.any(rock_areas):
    rock_overlay = np.zeros_like(img_color_rgb)
    rock_overlay[rock_areas] = [255, 100, 100]  # Light red for rocks
    ax1.imshow(rock_overlay, alpha=0.2)

# Plot all 3 paths
for i, (path, info) in enumerate(zip(all_paths, path_info)):
    path_coords = np.array(path)
    ax1.plot(path_coords[:, 1], path_coords[:, 0], 
             color=info['color'], linewidth=4, 
             label=f"{info['name']}", 
             alpha=0.9)

# Mark start points
for i, (start, info) in enumerate(zip(start_points[:len(all_paths)], path_info)):
    ax1.scatter(start[1], start[0], c=info['color'], s=200, marker='o', 
               edgecolors='white', linewidth=3, zorder=10)
    ax1.annotate(f'Start {i+1}', (start[1], start[0]), xytext=(10, 10), 
                textcoords='offset points', color='white', fontweight='bold', fontsize=10)

# Mark the horizon target
ax1.scatter(horizon_target[1], horizon_target[0], c='gold', s=300, marker='*', 
           label='Horizon Target', edgecolors='black', linewidth=3, zorder=10)
ax1.annotate('TARGET', (horizon_target[1], horizon_target[0]), xytext=(10, -15), 
            textcoords='offset points', color='black', fontweight='bold', fontsize=12)

ax1.set_title('3 Routes to Horizon Target\n(Sky areas shown in blue tint)', fontsize=16, fontweight='bold')
ax1.axis('off')
ax1.legend(loc='upper right', fontsize=12)

# Elevation map
ax2 = axes[0, 1]
elevation_display = ax2.imshow(elevation, cmap='terrain', alpha=0.9)
ax2.set_title('Elevation Map with Routes', fontsize=14, fontweight='bold')
plt.colorbar(elevation_display, ax=ax2, shrink=0.6)

# Plot paths on elevation map
for i, (path, info) in enumerate(zip(all_paths, path_info)):
    path_coords = np.array(path)
    ax2.plot(path_coords[:, 1], path_coords[:, 0], 
             color=info['color'], linewidth=3, alpha=0.8, label=info['name'])

ax2.scatter(horizon_target[1], horizon_target[0], c='gold', s=200, marker='*', 
           edgecolors='black', linewidth=2)
for i, start in enumerate(start_points[:len(all_paths)]):
    ax2.scatter(start[1], start[0], c=path_info[i]['color'], s=100, marker='o', 
               edgecolors='white', linewidth=2)

ax2.legend(fontsize=10)

# Steepness analysis
ax3 = axes[1, 0]
steepness_display = ax3.imshow(steepness, cmap='Reds', alpha=0.8)
ax3.set_title('Steepness Analysis', fontsize=14, fontweight='bold')
plt.colorbar(steepness_display, ax=ax3, shrink=0.6)

for path, info in zip(all_paths, path_info):
    path_coords = np.array(path)
    ax3.plot(path_coords[:, 1], path_coords[:, 0], 
             color=info['color'], linewidth=2, alpha=0.8)

ax3.scatter(horizon_target[1], horizon_target[0], c='gold', s=150, marker='*')

# Cost grid
ax4 = axes[1, 1]
cost_display_data = np.log(cost_grid + 1)
cost_display = ax4.imshow(cost_display_data, cmap='viridis_r', alpha=0.8)
ax4.set_title('Climbing Cost Grid (Log Scale)', fontsize=14, fontweight='bold')
plt.colorbar(cost_display, ax=ax4, shrink=0.6)

for path, info in zip(all_paths, path_info):
    path_coords = np.array(path)
    ax4.plot(path_coords[:, 1], path_coords[:, 0], 
             color=info['color'], linewidth=2, alpha=0.9)

ax4.scatter(horizon_target[1], horizon_target[0], c='gold', s=150, marker='*')

plt.tight_layout()
plt.show()

# Print detailed analysis
print("\n" + "="*80)
print("MOUNTAIN CLIMBING ROUTE ANALYSIS - HORIZON TARGET")
print("="*80)

target_elevation = elevation[horizon_target]
print(f"Horizon Target Location: {horizon_target}")
print(f"Target Elevation: {target_elevation:.3f}")

if path_info:
    print(f"\nFound {len(path_info)} routes to the horizon target:")
    
    for i, info in enumerate(path_info):
        print(f"\n{i+1}. {info['name']} ({info['color']} line):")
        print(f"   • Total Cost: {info['cost']:.1f}")
        print(f"   • Path Length: {info['path_length']} steps")
        print(f"   • Elevation Gain: {info['elevation_gain']:.3f}")
        print(f"   • Average Steepness: {info['avg_steepness']:.3f}")
        print(f"   • Rock Encounters: {info['rock_encounters']} sections")
        print(f"   • Final Elevation: {info['final_elevation']:.3f}")
        print(f"   • Reached Target: {'✓ YES' if info['reached_target'] else '✗ NO'}")
        print(f"   • Efficiency: {info['elevation_gain']/info['path_length']:.4f} (elevation/distance)")
    
    # Find the best route
    best_route = min(path_info, key=lambda x: x['cost'])
    print(f"\n🏆 BEST ROUTE: {best_route['name']} (lowest cost: {best_route['cost']:.1f})")
    
    # Check if all routes reach the target
    successful_routes = [info for info in path_info if info['reached_target']]
    print(f"\n📊 SUCCESS RATE: {len(successful_routes)}/{len(path_info)} routes successfully reach the horizon target")
else:
    print("❌ No valid paths found!")

print("\n" + "="*80)
