import numpy as np
import matplotlib.pyplot as plt

def main():

    def generate_intensity_map(shape=(200, 200), bg_intensity=10, ellipse1_intensity=50, ellipse2_intensity=20,
                               peak_intensity=50, bg_noise_std=2, peak_noise_std=8, ellipse2_noise_std=6, num_peaks=5, max_peak_radius=5):
        """
        Generates a map with background intensity, two elliptical regions with different intensities,
        added Gaussian noise (with different noise levels for background, second ellipse, and peaks), and small peaks of a given intensity.

        Parameters:
        shape (tuple): Dimensions of the map (height, width).
        bg_intensity (float): Background intensity value.
        ellipse1_intensity (float): Intensity of the first ellipse.
        ellipse2_intensity (float): Intensity of the second ellipse.
        peak_intensity (float): Intensity of the small peaks.
        bg_noise_std (float): Standard deviation of the Gaussian noise in the background.
        peak_noise_std (float): Standard deviation of the Gaussian noise in the peak regions.
        ellipse2_noise_std (float): Standard deviation of the Gaussian noise in the second ellipse.
        num_peaks (int): Number of small peaks to add.
        max_peak_radius (int): Maximum radius of the small peaks.

        Returns:
        numpy.ndarray: Intensity map with noise and peaks.
        """
        # Create the background
        intensity_map = np.full(shape, bg_intensity, dtype=float)

        # Define random scaling factors for the first ellipse
        random_scale_x1 = np.random.uniform(0.5, 1.5)  # Scale factor for x-axis
        random_scale_y1 = np.random.uniform(0.5, 1.5)  # Scale factor for y-axis
        random_rotation1 = np.random.uniform(0, 360)  # Random rotation angle in degrees

        # Define first ellipse parameters
        center = (shape[0] // 2, shape[1] // 2)  # Center of the ellipses
        axes1 = (int(random_scale_x1 * shape[0] // 4), int(random_scale_y1 * shape[1] // 6))  # Scaled axes for the first ellipse

        # Create a meshgrid for coordinates
        y, x = np.ogrid[:shape[0], :shape[1]]
        cos_angle1 = np.cos(np.radians(random_rotation1))
        sin_angle1 = np.sin(np.radians(random_rotation1))

        # Transform coordinates to align with the first ellipse's orientation
        x_rot1 = (x - center[1]) * cos_angle1 + (y - center[0]) * sin_angle1
        y_rot1 = -(x - center[1]) * sin_angle1 + (y - center[0]) * cos_angle1

        # Equation of the first ellipse
        ellipse1_region = (x_rot1 ** 2 / axes1[1] ** 2) + (y_rot1 ** 2 / axes1[0] ** 2) <= 1

        # Define random scaling factors for the second ellipse (independent of the first)
        random_scale_x2 = np.random.uniform(1.5, 2.5)  # Scale factor for x-axis (larger than the first)
        random_scale_y2 = np.random.uniform(1.5, 2.5)  # Scale factor for y-axis (larger than the first)
        random_rotation2 = np.random.uniform(0, 360)  # Random rotation angle in degrees for second ellipse

        # Define second ellipse parameters
        axes2 = (int(random_scale_x2 * shape[0] // 4), int(random_scale_y2 * shape[1] // 6))  # Scaled axes for the second ellipse

        cos_angle2 = np.cos(np.radians(random_rotation2))
        sin_angle2 = np.sin(np.radians(random_rotation2))

        # Transform coordinates to align with the second ellipse's orientation
        x_rot2 = (x - center[1]) * cos_angle2 + (y - center[0]) * sin_angle2
        y_rot2 = -(x - center[1]) * sin_angle2 + (y - center[0]) * cos_angle2

        # Equation of the second ellipse
        ellipse2_region = (x_rot2 ** 2 / axes2[1] ** 2) + (y_rot2 ** 2 / axes2[0] ** 2) <= 1

        # Set intensity for the ellipses
        intensity_map[ellipse2_region] = ellipse2_intensity  # Second ellipse intensity
        intensity_map[ellipse1_region] = ellipse1_intensity  # First ellipse intensity (overrides if they overlap)

        # Add small peaks inside the first ellipse
        for _ in range(num_peaks):
            # Random peak center within the first ellipse
            while True:
                peak_center = (np.random.randint(0, shape[0]), np.random.randint(0, shape[1]))
                if ellipse1_region[peak_center[0], peak_center[1]]:  # Check if the peak center is inside the first ellipse
                    break

            # Random peak radius
            peak_radius = np.random.randint(1, max_peak_radius + 1)

            # Create a circular mask for the peak
            y_peak, x_peak = np.ogrid[:shape[0], :shape[1]]
            peak_mask = (x_peak - peak_center[1]) ** 2 + (y_peak - peak_center[0]) ** 2 <= peak_radius ** 2

            # Add the peak intensity to the map
            intensity_map[peak_mask] = peak_intensity

        # Add Gaussian noise
        # Background noise
        bg_noise = np.random.normal(0, bg_noise_std, shape)
        intensity_map += bg_noise  # Add background noise to the whole map

        # Add noise specifically for the peak regions (regions with peaks)
        peak_noise = np.random.normal(0, peak_noise_std, shape)
        intensity_map[ellipse1_region] += peak_noise[ellipse1_region]  # Add noise only in the first ellipse (peaks region)

        # Add noise specifically for the second ellipse region
        ellipse2_noise = np.random.normal(0, ellipse2_noise_std, shape)
        intensity_map[ellipse2_region] += ellipse2_noise[ellipse2_region]  # Add noise only in the second ellipse region

        return intensity_map


    # Parameters
    shape = (200, 200)  # Map size
    bg_intensity = 0  # Background intensity
    ellipse1_intensity = 20  # Intensity of the first ellipse
    ellipse2_intensity = 10  # Intensity of the second ellipse
    peak_intensity = 25  # Intensity of the small peaks
    bg_noise_std = 2  # Standard deviation of background noise
    peak_noise_std = 8  # Standard deviation of noise in peak regions
    ellipse2_noise_std = 6  # Standard deviation of noise in the second ellipse region
    num_peaks = 500  # Number of peaks
    max_peak_radius = 5  # Maximum radius of the peaks

    # Generate the map
    intensity_map = generate_intensity_map(shape, bg_intensity, ellipse1_intensity, ellipse2_intensity, peak_intensity,
                                           bg_noise_std, peak_noise_std, ellipse2_noise_std, num_peaks, max_peak_radius)

    # Plot the map
    plt.figure(figsize=(8, 8))
    plt.imshow(intensity_map, cmap='gist_heat', origin='lower')
    plt.colorbar(label='Intensity')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(f"IMG_{np.random.rand(1,1)}.png")

main()


for i in range(20):
    main()